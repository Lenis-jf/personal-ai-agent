"""
Local MLX-VLM backend optimized for Gemma 4 with native Automatic Prefix Caching (APC),
Turn-0 static prefix warmup, VisionFeatureCache, and dynamic cache_trace.log inspection.
"""
from __future__ import annotations

from datetime import datetime
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator

# Ensure APC environment variables are set before MLX-VLM cache initialization
os.environ.setdefault("APC_ENABLED", "1")
os.environ.setdefault("APC_CHECKPOINT_ENTRIES", "32")
os.environ.setdefault("APC_CHECKPOINT_INTERVAL_TOKENS", "0")
os.environ.setdefault("APC_CHECKPOINT_GUARD_TOKENS", "1")
os.environ.setdefault("APC_DISK_ENABLED", "0")
os.environ.setdefault("APC_MEMORY_MAX_GB", "3.5")
os.environ.setdefault("APC_MEMORY_RESERVE_GB", "1.5")

import mlx.core as mx
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template
from PIL import Image

try:
    from mlx_vlm.apc import APCManager
except ImportError:
    APCManager = None

try:
    from mlx_vlm import VisionFeatureCache
except ImportError:
    VisionFeatureCache = None

from .base import LLM

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LOGS_DIR = PROJECT_ROOT / "logs"


class LocalLLMMLX(LLM):
    STOP_TOKENS = ("<" + "turn|>", "<" + "end_of_turn>", "<" + "eos>", "<" + "start_of_turn>", "<" + "tool_call|>")
    STOP_TOKEN_IDS = {1, 49, 106}  # 1 = '<eos>', 49 = '<tool_call|>', 106 = '<turn|>' in Gemma 4
    CHANNEL_OPEN_TOKEN_ID = 100    # 100 = '<|channel>' in Gemma 4
    CHANNEL_CLOSE_TOKEN_ID = 101   # 101 = '<channel|>' in Gemma 4
    PAD_TOKEN_ID = 0               # 0 = '<pad>' in Gemma 4
    PREFILL_STEP_SIZE = 4096

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model, self.processor = load(model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.image_token = getattr(self.processor, "image_token", "<|image|>")
        self.last_diagnostics: dict[str, Any] = {}
        self.cache_trace_path = LOGS_DIR / "cache_trace.log"

        self.apc_manager: Any = None
        self.vision_cache: Any = None
        self._static_prefix_prompt: str = ""
        self._init_cache_trace_log()
        self._init_caches()

    def _init_cache_trace_log(self) -> None:
        """Initializes logs/cache_trace.log to inspect what is cached and restored at every step."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            header = (
                f"{'=' * 92}\n"
                f"DYNAMIC KV CACHE & APC TRACE LOG (OVERWRITTEN AT SESSION START) — {ts}\n"
                f"Model: {self.model_path}\n"
                f"Purpose: Inspects exact token-level prefixes stored in and restored from KV Cache on each step.\n"
                f"{'=' * 92}\n\n"
            )
            self.cache_trace_path.write_text(header, encoding="utf-8")
        except Exception:
            pass

    def _append_cache_trace(self, block: str) -> None:
        """Appends a diagnostic cache inspection block to logs/cache_trace.log."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with self.cache_trace_path.open("a", encoding="utf-8") as f:
                f.write(block + "\n")
                f.flush()
        except Exception:
            pass

    def _record_step_cache_trace(
        self,
        step_label: str,
        formatted_prompt: str,
        use_apc: bool,
        apc_reason: str,
        has_images: bool,
        prompt_tokens: int | None,
        prompt_tps: float | None,
        ttft_sec: float | None = None,
    ) -> None:
        """
        Tokenizes the formatted prompt and logs the exact boundary between:
        1. Tokens restored from the KV cache (0 .. matched_tokens - 1)
        2. New suffix tokens prefilled by the GPU in this step (matched_tokens .. total_tokens)
        """
        try:
            token_ids: list[int] = []
            if hasattr(self.tokenizer, "encode"):
                token_ids = list(self.tokenizer.encode(formatted_prompt, add_special_tokens=True))

            total_tok = len(token_ids)
            stats = self.apc_manager.stats_snapshot() if (self.apc_manager is not None and use_apc and not has_images) else {}
            matched = int(stats.get("matched_tokens", 0)) if (use_apc and not has_images) else 0
            restored = int(stats.get("restored_tokens", 0)) if (use_apc and not has_images) else 0
            stored = int(stats.get("stored_tokens", 0)) if (use_apc and not has_images) else 0
            hit_rate = float(stats.get("token_hit_rate", 0.0)) * 100.0 if (use_apc and not has_images) else 0.0

            if has_images:
                mode_str = f"VISION_STEP (APC Bypassed for SigLIP2 | VisionFeatureCache=Active) — {apc_reason}"
            elif use_apc:
                mode_str = f"APC_WARM_PREFIX_CACHE — {apc_reason}"
            else:
                mode_str = f"HIGH_PRECISION_CLEAN_PREFILL (APC Bypassed) — {apc_reason}"

            first_5_tokens = [f"{tid}:{self._decode_single_token_debug(tid)}" for tid in token_ids[:5]]

            if matched > 0 and matched <= total_tok and hasattr(self.tokenizer, "decode"):
                cached_ids = token_ids[:matched]
                suffix_ids = token_ids[matched:]
                cached_text = self.tokenizer.decode(cached_ids, skip_special_tokens=False)
                suffix_text = self.tokenizer.decode(suffix_ids, skip_special_tokens=False)
                boundary_cached = [
                    f"{tid}:{self._decode_single_token_debug(tid)}"
                    for tid in cached_ids[max(0, len(cached_ids) - 4) :]
                ]
                boundary_suffix = [
                    f"{tid}:{self._decode_single_token_debug(tid)}"
                    for tid in suffix_ids[:4]
                ]
            else:
                cached_ids = []
                suffix_ids = token_ids
                cached_text = "(0 tokens restored from KV cache — full prompt prefilled fresh on GPU)"
                suffix_text = formatted_prompt
                boundary_cached = []
                boundary_suffix = [f"{tid}:{self._decode_single_token_debug(tid)}" for tid in token_ids[:4]]

            ttft_str = f"{ttft_sec:.2f}s" if ttft_sec is not None else "n/a"
            tps_str = f"{prompt_tps:.1f} tok/s" if prompt_tps else "n/a"

            block = (
                f"{'-' * 92}\n"
                f"[{datetime.now().strftime('%H:%M:%S')}] {step_label}\n"
                f"{'-' * 92}\n"
                f"• CACHE MODE       : {mode_str}\n"
                f"• TOKEN METRICS    : total_encoded={total_tok} (mlx_reported={prompt_tokens}) | "
                f"restored_from_cache={len(cached_ids)} tok | newly_prefilled_on_gpu={len(suffix_ids)} tok\n"
                f"• APC STATS        : matched={matched} | restored={restored} | newly_stored={stored} | "
                f"hit_rate={hit_rate:.1f}% | prefill_tps={tps_str} | TTFT={ttft_str}\n"
                f"• FIRST 5 TOKENS   : {first_5_tokens} (Verifies single <bos> ID 2 at index 0)\n"
                f"• BOUNDARY TOKENS  : last_cached={boundary_cached}  -->  first_new_suffix={boundary_suffix}\n\n"
                f"1. [RESTORED FROM KV CACHE ({len(cached_ids)} tokens)]:\n"
                f"   --- HEAD (first 250 chars) ---\n"
                f"   {repr(cached_text[:250])}\n"
                f"   --- TAIL (last 350 chars of cached prefix) ---\n"
                f"   {repr(cached_text[-350:]) if len(cached_text) > 250 else '(same as head)'}\n\n"
                f"2. [NEWLY PREFILLED SUFFIX SENT TO GPU ({len(suffix_ids)} tokens)]:\n"
                f"   --- HEAD (first 350 chars of new suffix) ---\n"
                f"   {repr(suffix_text[:350])}\n"
                f"   --- TAIL (last 350 chars of new suffix) ---\n"
                f"   {repr(suffix_text[-350:]) if len(suffix_text) > 350 else '(same as head)'}\n"
            )
            self._append_cache_trace(block)
        except Exception as exc:
            self._append_cache_trace(f"[CACHE TRACE ERROR on {step_label}]: {exc}")

    def _init_caches(self) -> None:
        """Initializes in-memory MLX-VLM Automatic Prefix Caching (APC) and VisionFeatureCache."""
        if APCManager is not None:
            try:
                self.apc_manager = APCManager(
                    num_blocks=4096,
                    block_size=16,
                    disk=None,
                )
            except Exception:
                self.apc_manager = None

        if VisionFeatureCache is not None:
            try:
                self.vision_cache = VisionFeatureCache()
            except Exception:
                self.vision_cache = None

    def warmup_static_prefix(self, static_system_prefix: str) -> None:
        """
        Pre-computes and pins the invariant System Prefix (base_prompt + core_profile + tools_prompt)
        into APC warm memory at Turn 0 without duplicating <bos>.
        """
        if not static_system_prefix or self.apc_manager is None:
            return
        self._static_prefix_prompt = static_system_prefix
        try:
            seed_prompt = f"<|turn>system\n{static_system_prefix.strip()}\n\n--- ACTIVE SKILL: "
            self.apc_manager.reset_stats()
            t0 = datetime.now()
            p_toks = None
            p_tps = None
            for chunk in stream_generate(
                model=self.model,
                processor=self.processor,
                prompt=seed_prompt,
                max_tokens=1,
                temperature=0.0,
                prefill_step_size=self.PREFILL_STEP_SIZE,
                apc_manager=self.apc_manager,
            ):
                if p_toks is None and hasattr(chunk, "prompt_tokens"):
                    p_toks = getattr(chunk, "prompt_tokens", None)
                if hasattr(chunk, "prompt_tps") and getattr(chunk, "prompt_tps", None):
                    p_tps = getattr(chunk, "prompt_tps", None)
            elapsed = (datetime.now() - t0).total_seconds()
            self._record_step_cache_trace(
                step_label="TURN 0 — STATIC SYSTEM PREFIX WARMUP (base_prompt + core_profile + tools_prompt)",
                formatted_prompt=seed_prompt,
                use_apc=True,
                apc_reason="Pinning invariant system prefix into APC warm RAM at session startup",
                has_images=False,
                prompt_tokens=p_toks,
                prompt_tps=p_tps,
                ttft_sec=elapsed,
            )
            self._clear_gpu_cache()
        except Exception:
            pass

    def rewarm_static_prefix(self) -> None:
        """Re-pins the Turn-0 static prefix if it was evicted."""
        if self._static_prefix_prompt:
            self.warmup_static_prefix(self._static_prefix_prompt)

    def clear_prefix_cache(self) -> None:
        """Closes and resets the APC manager, cache_trace.log, and re-warms the static Turn-0 prefix."""
        if self.apc_manager is not None:
            try:
                self.apc_manager.close()
            except Exception:
                pass
            self.apc_manager = None
        if self.vision_cache is not None and hasattr(self.vision_cache, "clear"):
            try:
                self.vision_cache.clear()
            except Exception:
                pass
        self._init_cache_trace_log()
        self._init_caches()
        if self._static_prefix_prompt:
            self.warmup_static_prefix(self._static_prefix_prompt)
        self._clear_gpu_cache()

    def get_special_tokens_summary(self) -> str:
        """Extracts special control tokens and their integer IDs from Gemma 4 tokenizer for logging."""
        lines = ["GEMMA 4 SPECIAL & CONTROL TOKENS MAP (ID -> Token):"]
        candidate_tokens = [
            "<pad>",
            "<eos>",
            "<bos>",
            "<unk>",
            "<" + "|turn>",
            "<" + "turn|>",
            "<" + "|channel>",
            "<" + "channel|>",
            "<" + "|tool_call>",
            "<" + "tool_call|>",
            "<" + "|tool_response>",
            "<" + "tool_response|>",
            "<" + "start_of_turn>",
            "<" + "end_of_turn>",
            "<thought>",
            "</thought>",
            "<" + "|image|>",
        ]
        seen_ids: set[int] = set()

        if hasattr(self.tokenizer, "convert_tokens_to_ids"):
            for tok_str in candidate_tokens:
                try:
                    tid = self.tokenizer.convert_tokens_to_ids(tok_str)
                    if isinstance(tid, int) and tid not in seen_ids:
                        seen_ids.add(tid)
                        lines.append(f"  • ID {tid:6d}  -->  {repr(tok_str)}")
                except Exception:
                    pass

        added_decoder = getattr(self.tokenizer, "added_tokens_decoder", None)
        if isinstance(added_decoder, dict):
            extra_count = 0
            for tid, tok_obj in sorted(added_decoder.items(), key=lambda x: int(x[0])):
                tid_int = int(tid)
                if tid_int in seen_ids:
                    continue
                content = getattr(tok_obj, "content", str(tok_obj))
                is_special = getattr(tok_obj, "special", True)
                if is_special:
                    seen_ids.add(tid_int)
                    lines.append(f"  • ID {tid_int:6d}  -->  {repr(content)} (special=True)")
                    extra_count += 1
                    if extra_count >= 25:
                        break

        return "\n".join(lines)

    @staticmethod
    def _prepare_image_inputs(
        images: list[str | Path | Image.Image] | None,
    ) -> tuple[list[str], list[str]]:
        """
        Validates images and ensures 3-channel RGB compatibility for Metal while returning
        file path strings so MLX-VLM's VisionFeatureCache can key entries by path.
        """
        if not images:
            return [], []

        prepared_paths: list[str] = []
        dimensions: list[str] = []
        tmp_dir = Path(tempfile.gettempdir()) / "personal_agent_rgb_cache"

        for idx, img_item in enumerate(images):
            try:
                if isinstance(img_item, (str, Path)):
                    p = Path(img_item).expanduser().resolve()
                    if not (p.exists() and p.is_file()):
                        continue
                    with Image.open(p) as opened_img:
                        dimensions.append(f"{opened_img.width}x{opened_img.height}")
                        if opened_img.mode == "RGB":
                            prepared_paths.append(str(p))
                        else:
                            tmp_dir.mkdir(parents=True, exist_ok=True)
                            rgb_path = tmp_dir / f"rgb_{p.stat().st_mtime_ns}_{p.stem}.png"
                            if not rgb_path.exists():
                                opened_img.convert("RGB").save(rgb_path, format="PNG")
                            prepared_paths.append(str(rgb_path))
                elif isinstance(img_item, Image.Image):
                    dimensions.append(f"{img_item.width}x{img_item.height}")
                    tmp_dir.mkdir(parents=True, exist_ok=True)
                    rgb_path = tmp_dir / f"pil_input_{idx}.png"
                    img_item.convert("RGB").save(rgb_path, format="PNG")
                    prepared_paths.append(str(rgb_path))
            except Exception:
                pass

        return prepared_paths, dimensions

    def _format_apc_status(
        self,
        has_images: bool,
        use_apc: bool = True,
        apc_reason: str = "",
        prompt_tokens: int | None = None,
        prompt_tps: float | None = None,
    ) -> str:
        """Formats APC and VisionFeatureCache telemetry for session_trace.log."""
        tps_info = f" | prompt_tokens={prompt_tokens} | prompt_tps={prompt_tps:.1f} tok/s" if prompt_tokens and prompt_tps else ""
        reason_info = f" [{apc_reason}]" if apc_reason else ""
        if has_images:
            v_status = "enabled" if self.vision_cache is not None else "unavailable"
            return f"vision_step (APC bypassed for SigLIP2, VisionFeatureCache={v_status}){reason_info}{tps_info}"
        if not use_apc:
            return f"HIGH_PRECISION_CLEAN_PREFILL (APC bypassed){reason_info}{tps_info}"
        if self.apc_manager is None:
            return f"APC unavailable (upgrade to mlx-vlm>=0.7.10){tps_info}"
        try:
            stats = self.apc_manager.stats_snapshot()
            matched = stats.get("matched_tokens", 0)
            restored = stats.get("restored_tokens", 0)
            stored = stats.get("stored_tokens", 0)
            hit_rate = float(stats.get("token_hit_rate", 0.0)) * 100.0
            disk_hits = stats.get("disk_hits", 0)
            return (
                f"APC active | matched={matched} tok | restored={restored} tok | "
                f"stored={stored} tok | hit_rate={hit_rate:.1f}% | disk_hits={disk_hits}{reason_info}{tps_info}"
            )
        except Exception as exc:
            return f"APC error ({exc}){tps_info}"

    def _format_input(
        self,
        prompt_or_messages: str | list[dict[str, str]],
        num_images: int = 0,
    ) -> tuple[str, str]:
        """
        Formats conversation messages into Gemma 4 chat template format, stripping any redundant
        leading <bos> string so processor.tokenizer.encode(add_special_tokens=True) adds a single <bos>.
        """
        image_prefix = (f"{self.image_token}\n" * num_images) if num_images > 0 else ""

        if isinstance(prompt_or_messages, list) and hasattr(self.tokenizer, "apply_chat_template"):
            normalized: list[dict[str, str]] = []
            for msg in prompt_or_messages:
                raw_role = str(msg.get("role", "user")).lower().strip()
                content = str(msg.get("content", "")).strip()
                if not content:
                    continue

                if raw_role == "system" and (not normalized or normalized[-1]["role"] == "system"):
                    role = "system"
                elif raw_role in {"assistant", "model"}:
                    role = "assistant"
                else:
                    role = "user"

                if normalized and normalized[-1]["role"] == role:
                    normalized[-1]["content"] += "\n\n" + content
                else:
                    normalized.append({"role": role, "content": content})

            for m in normalized:
                if self.image_token in m["content"] or "<|image|>" in m["content"]:
                    m["content"] = m["content"].replace(self.image_token, "").replace("<|image|>", "")

            if num_images > 0:
                for m in reversed(normalized):
                    if m["role"] == "user":
                        m["content"] = f"{image_prefix}{m['content']}"
                        break

            try:
                rendered = self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                bos_str = getattr(self.tokenizer, "bos_token", "<bos>") or "<bos>"
                if rendered.startswith(bos_str):
                    rendered = rendered[len(bos_str) :]
                mode_label = (
                    f"tokenizer.apply_chat_template(system_enabled, single_bos, images={num_images})"
                    if num_images > 0
                    else "tokenizer.apply_chat_template(system_enabled, single_bos)"
                )
                return rendered, mode_label
            except Exception as exc_primary:
                fallback_msgs: list[dict[str, str]] = []
                for m in normalized:
                    r = "user" if m["role"] == "system" else m["role"]
                    if fallback_msgs and fallback_msgs[-1]["role"] == r:
                        fallback_msgs[-1]["content"] += "\n\n" + m["content"]
                    else:
                        fallback_msgs.append({"role": r, "content": m["content"]})
                try:
                    rendered = self.tokenizer.apply_chat_template(
                        fallback_msgs,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                    bos_str = getattr(self.tokenizer, "bos_token", "<bos>") or "<bos>"
                    if rendered.startswith(bos_str):
                        rendered = rendered[len(bos_str) :]
                    return rendered, f"tokenizer.apply_chat_template(fallback_user_only: {exc_primary})"
                except Exception:
                    pass

        prompt_str = (
            prompt_or_messages
            if isinstance(prompt_or_messages, str)
            else "\n\n".join(f"{m['role']}: {m['content']}" for m in prompt_or_messages)
        )
        rendered_fb = apply_chat_template(self.processor, self.model.config, prompt_str, num_images=num_images)
        bos_str = getattr(self.tokenizer, "bos_token", "<bos>") or "<bos>"
        if isinstance(rendered_fb, str) and rendered_fb.startswith(bos_str):
            rendered_fb = rendered_fb[len(bos_str) :]
        return rendered_fb, "mlx_vlm.apply_chat_template"

    def _clear_gpu_cache(self) -> None:
        try:
            mx.clear_cache()
        except Exception:
            pass

    def _decode_single_token_debug(self, tok_id: int) -> str:
        """Decodes an individual token with skip_special_tokens=False for trace debugging."""
        if hasattr(self.tokenizer, "decode"):
            try:
                return repr(self.tokenizer.decode([tok_id], skip_special_tokens=False))
            except Exception:
                pass
        return str(tok_id)

    @staticmethod
    def _has_token_repetition_loop(token_ids: list[int]) -> bool:
        """Detects repetitive token loops to abort degenerate generation cycles early."""
        n = len(token_ids)
        if n < 18:
            return False
        for pat_len in (1, 2, 3, 4):
            if n >= pat_len * 6:
                pattern = token_ids[-pat_len:]
                if token_ids[-pat_len * 6 :] == pattern * 6:
                    return True
        return False

    def generate(
        self,
        prompt: str | list[dict[str, str]],
        images: list[str | Path | Image.Image] | None = None,
        max_tokens: int = 2048,
        use_apc: bool = True,
        apc_reason: str = "standard",
        step_label: str = "GENERATE",
    ) -> str:
        image_paths, image_dims = self._prepare_image_inputs(images)
        formatted_prompt, template_method = self._format_input(prompt, num_images=len(image_paths))

        self.last_diagnostics = {
            "template_method": template_method,
            "formatted_prompt": formatted_prompt,
            "kv_cache_status": "pending",
            "stop_reason": "generate_complete",
            "total_chunks": 0,
            "buffered_subword_chunks": 0,
            "buffered_tokens_sample": [],
            "raw_with_special_tokens": "",
            "token_id_trace": [],
            "images_count": len(image_paths),
            "images_dimensions": image_dims,
        }

        extra_kwargs: dict[str, Any] = {}
        if image_paths:
            extra_kwargs["image"] = image_paths
            if self.vision_cache is not None:
                extra_kwargs["vision_cache"] = self.vision_cache
        elif use_apc and self.apc_manager is not None:
            try:
                self.apc_manager.reset_stats()
                extra_kwargs["apc_manager"] = self.apc_manager
            except Exception:
                pass

        try:
            result = generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                prefill_step_size=self.PREFILL_STEP_SIZE,
                **extra_kwargs,
            )
            text = result.text if hasattr(result, "text") else str(result)
            p_toks = getattr(result, "prompt_tokens", None)
            p_tps = getattr(result, "prompt_tps", None)
            self.last_diagnostics["kv_cache_status"] = self._format_apc_status(
                has_images=bool(image_paths),
                use_apc=use_apc,
                apc_reason=apc_reason,
                prompt_tokens=p_toks,
                prompt_tps=p_tps,
            )
            self._record_step_cache_trace(
                step_label=step_label,
                formatted_prompt=formatted_prompt,
                use_apc=use_apc,
                apc_reason=apc_reason,
                has_images=bool(image_paths),
                prompt_tokens=p_toks,
                prompt_tps=p_tps,
            )
            self.last_diagnostics["raw_with_special_tokens"] = text
            for stop_tok in self.STOP_TOKENS:
                if stop_tok in text:
                    text = text.split(stop_tok, 1)[0]
                    self.last_diagnostics["stop_reason"] = f"stop_token:{stop_tok}"
            return text
        finally:
            self._clear_gpu_cache()

    def stream(
        self,
        prompt: str | list[dict[str, str]],
        images: list[str | Path | Image.Image] | None = None,
        max_tokens: int = 2048,
        use_apc: bool = True,
        apc_reason: str = "standard",
        step_label: str = "STREAM_STEP",
    ) -> Iterator[str]:
        image_paths, image_dims = self._prepare_image_inputs(images)
        formatted_prompt, template_method = self._format_input(prompt, num_images=len(image_paths))

        pad_streak = 0
        buffered_subword_chunks = 0
        total_chunks = 0
        recent_token_ids: list[int] = []
        generated_token_ids: list[int] = []
        buffered_tokens_sample: list[str] = []
        yielded_pieces: list[str] = []
        stop_reason = "natural_eos_or_max_tokens"
        prompt_tokens: int | None = None
        prompt_tps: float | None = None
        substantial_answer_started = False
        accumulated_text = ""

        self.last_diagnostics = {
            "template_method": template_method,
            "formatted_prompt": formatted_prompt,
            "kv_cache_status": "pending",
            "stop_reason": stop_reason,
            "total_chunks": 0,
            "buffered_subword_chunks": 0,
            "buffered_tokens_sample": [],
            "raw_with_special_tokens": "",
            "token_id_trace": [],
            "images_count": len(image_paths),
            "images_dimensions": image_dims,
        }

        extra_kwargs: dict[str, Any] = {}
        if image_paths:
            extra_kwargs["image"] = image_paths
            if self.vision_cache is not None:
                extra_kwargs["vision_cache"] = self.vision_cache
        elif use_apc and self.apc_manager is not None:
            try:
                self.apc_manager.reset_stats()
                extra_kwargs["apc_manager"] = self.apc_manager
            except Exception:
                pass

        t_stream_start = datetime.now()
        t_first_tok_sec: float | None = None

        try:
            for chunk in stream_generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
                prefill_step_size=self.PREFILL_STEP_SIZE,
                **extra_kwargs,
            ):
                total_chunks += 1
                if t_first_tok_sec is None:
                    t_first_tok_sec = (datetime.now() - t_stream_start).total_seconds()
                if prompt_tokens is None and hasattr(chunk, "prompt_tokens"):
                    prompt_tokens = getattr(chunk, "prompt_tokens", None)
                if hasattr(chunk, "prompt_tps") and getattr(chunk, "prompt_tps", None):
                    prompt_tps = getattr(chunk, "prompt_tps", None)

                raw_tok = getattr(chunk, "token", None)
                tok_id: int | None = None
                if raw_tok is not None:
                    tok_id = int(raw_tok.item()) if hasattr(raw_tok, "item") else int(raw_tok)

                    # Only trigger the late channel token firewall if a real thought block already
                    # closed AND the model has already emitted substantial natural-language answer text (>80 chars)
                    if substantial_answer_started and tok_id in {self.CHANNEL_OPEN_TOKEN_ID, self.CHANNEL_CLOSE_TOKEN_ID}:
                        stop_reason = f"late_channel_token_firewall(id={tok_id})"
                        break

                    generated_token_ids.append(tok_id)
                    recent_token_ids.append(tok_id)
                    if len(recent_token_ids) > 36:
                        recent_token_ids.pop(0)

                    if self._has_token_repetition_loop(recent_token_ids):
                        stop_reason = f"token_repetition_loop(recent={recent_token_ids[-12:]})"
                        break

                    if tok_id == self.PAD_TOKEN_ID:
                        pad_streak += 1
                        if pad_streak >= 16:
                            stop_reason = "pad_token_streak_16"
                            break
                    else:
                        pad_streak = 0

                # Rely natively on chunk.text for <|channel> (100) and <channel|> (101) since they are
                # non-special tokens in Gemma 4 and are emitted in exact word-order by the BPE detokenizer
                piece = chunk.text if hasattr(chunk, "text") else str(chunk)

                if substantial_answer_started and "<channel|>" in piece:
                    before_ch = piece.split("<channel|>", 1)[0]
                    if before_ch:
                        yielded_pieces.append(before_ch)
                        yield before_ch
                    stop_reason = "late_channel_string_firewall"
                    break

                if not piece:
                    buffered_subword_chunks += 1
                    if tok_id is not None and len(buffered_tokens_sample) < 30:
                        buffered_tokens_sample.append(f"{tok_id}:{self._decode_single_token_debug(tok_id)}")
                    if tok_id in self.STOP_TOKEN_IDS:
                        stop_reason = f"stop_token_id:{tok_id}"
                        break
                    continue

                if any(stop_tok in piece for stop_tok in self.STOP_TOKENS):
                    matched_tok = next(t for t in self.STOP_TOKENS if t in piece)
                    cut_idx = min(piece.find(t) for t in self.STOP_TOKENS if t in piece)
                    before_stop = piece[:cut_idx]
                    if before_stop:
                        yielded_pieces.append(before_stop)
                        yield before_stop
                    stop_reason = f"stop_token:{matched_tok}"
                    break

                accumulated_text += piece
                if not substantial_answer_started and ("</thought>" in accumulated_text or "<channel|>" in accumulated_text):
                    norm_acc = accumulated_text.replace("<channel|>", "</thought>")
                    parts = norm_acc.split("</thought>")
                    after_last_close = parts[-1].strip()
                    if (
                        len(after_last_close) >= 80
                        and not after_last_close.startswith(("{", "<thought>", "<|channel>", "call:"))
                    ):
                        substantial_answer_started = True

                yielded_pieces.append(piece)
                yield piece

                if tok_id in self.STOP_TOKEN_IDS:
                    stop_reason = f"stop_token_id:{tok_id}"
                    break

            if generated_token_ids and hasattr(self.tokenizer, "decode") and not stop_reason.startswith("late_channel"):
                try:
                    full_clean = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    for stop_tok in (*self.STOP_TOKENS, "<channel|>"):
                        if stop_tok in full_clean:
                            full_clean = full_clean.split(stop_tok, 1)[0]
                    already_emitted = "".join(yielded_pieces)
                    if len(full_clean) > len(already_emitted) and full_clean.startswith(already_emitted):
                        unflushed_tail = full_clean[len(already_emitted) :]
                        if unflushed_tail:
                            yielded_pieces.append(unflushed_tail)
                            yield unflushed_tail
                except Exception:
                    pass
        finally:
            raw_with_special = ""
            if generated_token_ids and hasattr(self.tokenizer, "decode"):
                try:
                    raw_with_special = self.tokenizer.decode(generated_token_ids, skip_special_tokens=False)
                except Exception:
                    raw_with_special = "<decode_error>"

            if len(generated_token_ids) <= 60:
                trace_items = [
                    f"{tid}:{self._decode_single_token_debug(tid)}" for tid in generated_token_ids
                ]
            else:
                head_ids = [f"{tid}:{self._decode_single_token_debug(tid)}" for tid in generated_token_ids[:30]]
                tail_ids = [f"{tid}:{self._decode_single_token_debug(tid)}" for tid in generated_token_ids[-30:]]
                trace_items = head_ids + [f"... ({len(generated_token_ids) - 60} intermediate tokens) ..."] + tail_ids

            apc_status_str = self._format_apc_status(
                has_images=bool(image_paths),
                use_apc=use_apc,
                apc_reason=apc_reason,
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tps,
            )

            self._record_step_cache_trace(
                step_label=step_label,
                formatted_prompt=formatted_prompt,
                use_apc=use_apc,
                apc_reason=apc_reason,
                has_images=bool(image_paths),
                prompt_tokens=prompt_tokens,
                prompt_tps=prompt_tps,
                ttft_sec=t_first_tok_sec,
            )

            self.last_diagnostics.update({
                "kv_cache_status": apc_status_str,
                "stop_reason": stop_reason,
                "total_chunks": total_chunks,
                "buffered_subword_chunks": buffered_subword_chunks,
                "buffered_tokens_sample": buffered_tokens_sample,
                "raw_with_special_tokens": raw_with_special,
                "token_id_trace": trace_items,
            })

            self._clear_gpu_cache()