"""Backend local MLX-VLM optimizado para Gemma 4 con tabla de tokens especiales y trazabilidad completa."""
from __future__ import annotations

from typing import Any, Iterator
import mlx.core as mx
from mlx_vlm import load, generate, stream_generate
from mlx_vlm.prompt_utils import apply_chat_template

from .base import LLM


class LocalLLMMLX(LLM):
    STOP_TOKENS = ("<" + "turn|>", "<" + "end_of_turn>", "<" + "eos>", "<" + "start_of_turn>")
    STOP_TOKEN_IDS = {1, 106}  # 1 = '<eos>', 106 = '<turn|>' en Gemma 4
    PAD_TOKEN_ID = 0           # 0 = '<pad>' en Gemma 4

    def __init__(self, model_path: str) -> None:
        self.model_path = model_path
        self.model, self.processor = load(model_path)
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.last_diagnostics: dict[str, Any] = {}

    def get_special_tokens_summary(self) -> str:
        """
        Extrae del tokenizador de Gemma 4 la lista de tokens especiales y sus IDs numéricos
        para documentarlos al inicio de logs/session_trace.log.
        """
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

    def _format_input(self, prompt_or_messages: str | list[dict[str, str]]) -> tuple[str, str]:
        """
        Formatea la lista de mensajes preservando el rol 'system' inicial nativo de Gemma 4
        y devuelve (formatted_prompt, template_method_used) para el log de diagnóstico.
        """
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

            try:
                rendered = self.tokenizer.apply_chat_template(
                    normalized,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return rendered, "tokenizer.apply_chat_template(system_enabled)"
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
                    return rendered, f"tokenizer.apply_chat_template(fallback_user_only: {exc_primary})"
                except Exception:
                    pass

        prompt_str = (
            prompt_or_messages
            if isinstance(prompt_or_messages, str)
            else "\n\n".join(f"{m['role']}: {m['content']}" for m in prompt_or_messages)
        )
        return apply_chat_template(self.processor, self.model.config, prompt_str), "mlx_vlm.apply_chat_template"

    def _clear_gpu_cache(self) -> None:
        try:
            mx.clear_cache()
        except Exception:
            pass

    def _decode_single_token_debug(self, tok_id: int) -> str:
        """Decodifica un token individual con skip_special_tokens=False para ver tokens de control ocultos."""
        if hasattr(self.tokenizer, "decode"):
            try:
                return repr(self.tokenizer.decode([tok_id], skip_special_tokens=False))
            except Exception:
                pass
        return str(tok_id)

    @staticmethod
    def _has_token_repetition_loop(token_ids: list[int]) -> bool:
        """Corta de inmediato si la GPU repite el mismo patrón corto de tokens 6 veces seguidas."""
        n = len(token_ids)
        if n < 18:
            return False
        for pat_len in (1, 2, 3, 4):
            if n >= pat_len * 6:
                pattern = token_ids[-pat_len:]
                if token_ids[-pat_len * 6 :] == pattern * 6:
                    return True
        return False

    def generate(self, prompt: str | list[dict[str, str]], max_tokens: int = 2048) -> str:
        formatted_prompt, template_method = self._format_input(prompt)
        self.last_diagnostics = {
            "template_method": template_method,
            "formatted_prompt": formatted_prompt,
            "stop_reason": "generate_complete",
            "total_chunks": 0,
            "buffered_subword_chunks": 0,
            "buffered_tokens_sample": [],
            "raw_with_special_tokens": "",
            "token_id_trace": [],
        }
        try:
            result = generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            )
            text = result.text if hasattr(result, "text") else str(result)
            self.last_diagnostics["raw_with_special_tokens"] = text
            for stop_tok in self.STOP_TOKENS:
                if stop_tok in text:
                    text = text.split(stop_tok, 1)[0]
                    self.last_diagnostics["stop_reason"] = f"stop_token:{stop_tok}"
            return text
        finally:
            self._clear_gpu_cache()

    def stream(self, prompt: str | list[dict[str, str]], max_tokens: int = 2048) -> Iterator[str]:
        formatted_prompt, template_method = self._format_input(prompt)
        pad_streak = 0
        buffered_subword_chunks = 0
        total_chunks = 0
        recent_token_ids: list[int] = []
        generated_token_ids: list[int] = []
        buffered_tokens_sample: list[str] = []
        yielded_pieces: list[str] = []
        stop_reason = "natural_eos_or_max_tokens"

        self.last_diagnostics = {
            "template_method": template_method,
            "formatted_prompt": formatted_prompt,
            "stop_reason": stop_reason,
            "total_chunks": 0,
            "buffered_subword_chunks": 0,
            "buffered_tokens_sample": [],
            "raw_with_special_tokens": "",
            "token_id_trace": [],
        }

        try:
            for chunk in stream_generate(
                model=self.model,
                processor=self.processor,
                prompt=formatted_prompt,
                max_tokens=max_tokens,
                temperature=0.0,
            ):
                total_chunks += 1
                raw_tok = getattr(chunk, "token", None)
                tok_id: int | None = None
                if raw_tok is not None:
                    tok_id = int(raw_tok.item()) if hasattr(raw_tok, "item") else int(raw_tok)
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

                piece = chunk.text if hasattr(chunk, "text") else str(chunk)

                # En mlx_vlm, chunk.text viene vacío ("") cuando el detokenizador agrupa sub-tokens
                # sin espacios intermedios (ej. rutas largas o UUIDs). No debemos cortar ahí.
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

                yielded_pieces.append(piece)
                yield piece

                if tok_id in self.STOP_TOKEN_IDS:
                    stop_reason = f"stop_token_id:{tok_id}"
                    break

            # Si el detokenizador de mlx_vlm retuvo sub-tokens finales sin espacio al cerrar el stream,
            # comparamos con la decodificación directa de generated_token_ids y vaciamos el resto.
            if generated_token_ids and hasattr(self.tokenizer, "decode"):
                try:
                    full_clean = self.tokenizer.decode(generated_token_ids, skip_special_tokens=True)
                    for stop_tok in self.STOP_TOKENS:
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

            self.last_diagnostics.update({
                "stop_reason": stop_reason,
                "total_chunks": total_chunks,
                "buffered_subword_chunks": buffered_subword_chunks,
                "buffered_tokens_sample": buffered_tokens_sample,
                "raw_with_special_tokens": raw_with_special,
                "token_id_trace": trace_items,
            })
            self._clear_gpu_cache()