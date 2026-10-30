"""
Agent orchestrator managing native multi-turn conversations, adaptive two-stage
Intern-Decision routing, Gemma 4 multimodal vision, Smart Hybrid APC, and session logging.
"""
from __future__ import annotations

from datetime import datetime
import json
import re
import time
from pathlib import Path
from typing import Any, Callable

from intern_decision_mlx import Decider
from llm.base import LLM
from router.decision import RoutingDecision
from router.schema import focused_subskill_schema, question_schema
from tools.file_tools import TOOL_METADATA

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
SKILLS_DIR = ROOT_DIR / "skills"
MODULES_DIR = SKILLS_DIR / "modules"
CONTEXT_DIR = ROOT_DIR / "context"
LOGS_DIR = PROJECT_ROOT / "logs"

FILE_TOOLKIT = {"file_search", "dir_tree", "file_grep", "file_read"}

SKILL_TOOL_BUNDLES: dict[str, set[str]] = {
    "file_management": FILE_TOOLKIT,
    "coding": FILE_TOOLKIT,
}

MODULE_ORDER = (
    "file_locator",
    "document_reader",
    "code_inspector",
    "content_grep",
    "image_vision",
)

VISION_MARKER_REGEX = re.compile(
    r"\[(?:IMAGE_READY_FOR_VISUALIZATION|IMAGEN_LISTA_PARA_VISION):\s*([^\]]+)\]"
)


class Agent:
    def __init__(self, llm: LLM, decider: Decider | None = None, max_steps: int = 6) -> None:
        self.llm = llm
        self.decider = decider if decider is not None else Decider()
        self.tool_registry = TOOL_METADATA
        self.max_steps = max_steps
        self.turn_counter = 0
        self.trace_log_path = LOGS_DIR / "session_trace.log"
        self.history_log_path = LOGS_DIR / "session_history.md"
        self.reset_session_logs()
        self._warmup_turn_zero_prefix()

    def _warmup_turn_zero_prefix(self) -> None:
        """Pre-warms the invariant System Prefix (base_prompt + core_profile + tools_prompt) into APC at Turn 0."""
        if hasattr(self.llm, "warmup_static_prefix"):
            try:
                base_prompt = self._load_base_prompt()
                core_profile = self._load_core_profile()
                tools_prompt = self._format_tools_description(FILE_TOOLKIT)
                static_prefix = (
                    f"{base_prompt}\n\n"
                    f"{core_profile}\n\n"
                    f"--- RUNTIME TOOLS ---\n"
                    f"{tools_prompt}"
                )
                self.llm.warmup_static_prefix(static_prefix)
            except Exception:
                pass

    def reset_session_logs(self) -> None:
        """Initializes blank session trace and history logs on session startup or clear command."""
        self.turn_counter = 0
        if hasattr(self.llm, "clear_prefix_cache"):
            try:
                self.llm.clear_prefix_cache()
            except Exception:
                pass

        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            model_name = getattr(self.llm, "model_path", "unknown_model")
            special_tokens_info = (
                self.llm.get_special_tokens_summary()
                if hasattr(self.llm, "get_special_tokens_summary")
                else "(No special tokens table available)"
            )
            trace_header = (
                f"{'=' * 88}\n"
                f"SESSION TRACE LOG (OVERWRITTEN ONLY AT SESSION START) — {ts}\n"
                f"Model: {model_name}\n"
                f"{'=' * 88}\n"
                f"{special_tokens_info}\n"
                f"{'=' * 88}\n\n"
            )
            self.trace_log_path.write_text(trace_header, encoding="utf-8")

            history_header = (
                f"# Session Conversation & Tool Memory History\n"
                f"- **Session Started**: {ts}\n"
                f"- **Model**: `{model_name}`\n"
                f"- **Lifecycle**: Initialized at session start; appends each turn sequentially.\n\n"
                f"---\n\n"
            )
            self.history_log_path.write_text(history_header, encoding="utf-8")
        except Exception:
            pass

    def _append_trace_log(self, block: str) -> None:
        """Appends an execution trace block to session_trace.log with immediate disk flush."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with self.trace_log_path.open("a", encoding="utf-8") as f:
                f.write(block + "\n")
                f.flush()
        except Exception:
            pass

    @staticmethod
    def _extract_vision_paths(tool_output: str) -> list[str]:
        """Extracts file paths flagged by the [IMAGE_READY_FOR_VISUALIZATION: <path>] marker."""
        matches = VISION_MARKER_REGEX.findall(tool_output)
        return [m.strip() for m in matches if m.strip()]

    @staticmethod
    def _strip_vision_markers(tool_output: str) -> str:
        """Removes visual pipeline markers from Tool Memory before archiving in conversation history."""
        return VISION_MARKER_REGEX.sub("", tool_output).replace("\n\n\n", "\n\n").strip()

    def append_turn_history_log(
        self,
        turn_num: int,
        user_message: str,
        final_response: str,
        stored_memories: list[dict[str, str]],
        injected_history_snapshot: str,
        mem_stats: dict[str, int],
        total_history_messages: int,
    ) -> None:
        """Appends the complete record of Turn N to logs/session_history.md."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H:%M:%S")
            lines = [
                f"## Turn {turn_num} — [{ts}]",
                f"- **Total messages in `history` at turn end**: {total_history_messages} "
                f"({total_history_messages // 2} completed turns)",
                f"- **Previous turn memories injected at Turn {turn_num} start**: "
                f"Full={mem_stats.get('full_memories', 0)} | Compacted={mem_stats.get('compacted_memories', 0)}\n",
                f"### 1. Previous Session Context Read by Model at Turn {turn_num} Start:",
                "----------------------------------------",
                injected_history_snapshot.strip() if injected_history_snapshot.strip() else "(Initial turn — no prior history)",
                "----------------------------------------\n",
                f"### 2. User Request (`Juan`) in Turn {turn_num}:",
                "----------------------------------------",
                user_message.strip(),
                "----------------------------------------\n",
            ]

            if stored_memories:
                lines.append(f"### 3. New `[Tool Memory]` Blocks Saved in Turn {turn_num} ({len(stored_memories)}):")
                for idx, tm in enumerate(stored_memories, start=1):
                    t_name = tm.get("tool", "tool")
                    t_out = str(tm.get("output", "")).strip()
                    lines.append(f"**3.{idx} Tool `{t_name}` ({len(t_out)} chars):**")
                    lines.append("----------------------------------------")
                    lines.append(t_out)
                    lines.append("----------------------------------------")
                lines.append("")
            else:
                lines.append(
                    f"### 3. New `[Tool Memory]` Blocks Saved in Turn {turn_num}:\n"
                    "*(No tools executed in this turn — model answered directly from session memory)*\n"
                )

            lines.extend([
                f"### 4. Final `Assistant` Response Saved in Memory (Turn {turn_num}):",
                "----------------------------------------",
                final_response.strip(),
                "----------------------------------------\n",
                "---\n\n",
            ])

            with self.history_log_path.open("a", encoding="utf-8") as f:
                f.write("\n".join(lines))
                f.flush()
        except Exception:
            pass

    def decide_route(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
    ) -> RoutingDecision:
        """
        Two-stage intelligent routing with Intern-Decision:
        - Stage 1: Evaluates user_message in isolation to prevent topic drag on new topics.
        - Stage 2 (Context-driven re-routing): Triggered if Intern-Decision identifies that
          the request depends on earlier turns (needs_previous_context=True), if skill
          confidence is low (<0.55), or if zero sub-modules exceeded threshold for an operational task.
        """
        try:
            stage1_req = {
                "state": {"user": user_message},
                "questions": question_schema("en"),
            }
            raw_stage1 = self.decider.predict(stage1_req)
            decision = RoutingDecision.from_answers(raw_stage1["answers"])

            needs_reroute = (
                (decision.needs_previous_context and bool(history))
                or decision.skill_confidence < 0.55
                or (decision.skill in {"file_management", "coding"} and not decision.sub_skills)
            )

            if needs_reroute:
                recent_context_lines: list[str] = []
                if history:
                    for turn in history[-2:]:
                        r_label = "Previous Assistant Response" if turn.get("role") == "assistant" else "Previous User Request"
                        snippet = str(turn.get("content", "")).strip()[:280]
                        t_mems = turn.get("tool_memories") or []
                        tools_used = [tm.get("tool", "") for tm in t_mems if tm.get("tool")]
                        tool_note = f" (Tools used: {tools_used})" if tools_used else ""
                        if snippet:
                            recent_context_lines.append(f"- {r_label}{tool_note}: {snippet}")

                enriched_user_state = (
                    "[Previous Turn Summary]\n" + "\n".join(recent_context_lines) + f"\n\n[Current User Request]\n{user_message}"
                    if recent_context_lines
                    else user_message
                )

                stage2_req = {
                    "state": {"user": enriched_user_state},
                    "questions": focused_subskill_schema(decision.skill),
                }
                raw_stage2 = self.decider.predict(stage2_req)
                stage2_dec = RoutingDecision.from_answers(
                    raw_stage2["answers"],
                    sub_skill_threshold=0.40,
                    tool_threshold=0.42,
                    rerouted=True,
                )

                chosen_skill = stage2_dec.skill if stage2_dec.skill_confidence >= decision.skill_confidence else decision.skill
                merged_tools = frozenset(t for t in (decision.tools | stage2_dec.tools) if t in self.tool_registry)
                merged_subs = decision.sub_skills | stage2_dec.sub_skills
                merged_confidences = {**decision.sub_skill_confidences, **stage2_dec.sub_skill_confidences}
                merged_probs = {**decision.probabilities, **stage2_dec.probabilities}

                return RoutingDecision(
                    skill=chosen_skill,
                    complexity=decision.complexity,
                    model_route=decision.model_route,
                    tools=merged_tools,
                    needs_clarification=stage2_dec.needs_clarification,
                    requires_confirmation=stage2_dec.requires_confirmation,
                    probabilities=merged_probs,
                    sub_skills=merged_subs,
                    sub_skill_confidences=merged_confidences,
                    needs_previous_context=decision.needs_previous_context,
                    skill_confidence=max(decision.skill_confidence, stage2_dec.skill_confidence),
                    rerouted=True,
                )

            filtered_tools = frozenset(t for t in decision.tools if t in self.tool_registry)
            return RoutingDecision(
                skill=decision.skill,
                complexity=decision.complexity,
                model_route=decision.model_route,
                tools=filtered_tools,
                needs_clarification=decision.needs_clarification,
                requires_confirmation=decision.requires_confirmation,
                probabilities=decision.probabilities,
                sub_skills=decision.sub_skills,
                sub_skill_confidences=decision.sub_skill_confidences,
                needs_previous_context=decision.needs_previous_context,
                skill_confidence=decision.skill_confidence,
                rerouted=False,
            )
        except Exception:
            return RoutingDecision(
                skill="file_management",
                complexity="moderate",
                model_route="local",
                tools=frozenset(FILE_TOOLKIT),
                needs_clarification=False,
                requires_confirmation=False,
                probabilities={},
                sub_skills=frozenset({"file_locator", "document_reader"}),
                sub_skill_confidences={"file_locator": 0.50, "document_reader": 0.50},
                needs_previous_context=False,
                skill_confidence=0.0,
                rerouted=False,
            )

    def _resolve_active_tools(self, decision: RoutingDecision) -> set[str]:
        """Resolves the set of authorized tools enabled for the current decision."""
        active = {t for t in decision.tools if t in self.tool_registry}
        if decision.skill in SKILL_TOOL_BUNDLES:
            active.update(SKILL_TOOL_BUNDLES[decision.skill])
        if active & FILE_TOOLKIT:
            active.update(FILE_TOOLKIT)
        return active

    def _load_base_prompt(self) -> str:
        base_file = CONTEXT_DIR / "base_prompt.md"
        return base_file.read_text(encoding="utf-8") if base_file.exists() else ""

    def _load_core_profile(self) -> str:
        profile_file = CONTEXT_DIR / "core_profile.md"
        return profile_file.read_text(encoding="utf-8") if profile_file.exists() else ""

    def _compose_modular_skill(
        self,
        decision: RoutingDecision,
        active_tools: set[str],
    ) -> tuple[str, str]:
        """
        Dynamically composes the base skill alongside the modular sub-skills selected by
        Intern-Decision, formatting individual confidence scores for downstream LLM evaluation.
        """
        skill_name = decision.skill
        base_skill_file = SKILLS_DIR / f"{skill_name}.md"
        if not base_skill_file.exists():
            fallback_name = "file_management" if active_tools else "personal_assistant"
            base_skill_file = SKILLS_DIR / f"{fallback_name}.md"
            skill_name = fallback_name

        loaded_labels: list[str] = []
        sections: list[str] = []

        if base_skill_file.exists():
            sections.append(base_skill_file.read_text(encoding="utf-8").strip())
            loaded_labels.append(base_skill_file.name)
        else:
            sections.append("You are a helpful, direct, and grounded personal AI assistant for Juan Fernando.")
            loaded_labels.append("builtin_default")

        for mod_name in MODULE_ORDER:
            if mod_name in decision.sub_skills:
                mod_path = MODULES_DIR / f"{mod_name}.md"
                if mod_path.exists():
                    sections.append(mod_path.read_text(encoding="utf-8").strip())
                    loaded_labels.append(f"modules/{mod_name}.md")

        recommended_tools = sorted(list(decision.tools)) if decision.tools else sorted(list(active_tools))

        sub_skills_bullets: list[str] = []
        for mod in sorted(list(decision.sub_skills)):
            conf = decision.sub_skill_confidences.get(mod)
            if conf is not None:
                tier = "high" if conf >= 0.70 else ("moderate" if conf >= 0.55 else "advisory - evaluate relevance autonomously")
                sub_skills_bullets.append(f"  - `{mod}` (relevance tier: {tier})")
            else:
                sub_skills_bullets.append(f"  - `{mod}`")

        sub_skills_rendered = "\n".join(sub_skills_bullets) if sub_skills_bullets else "  - (No sub-skills required for this turn)"
        skill_tier = "high" if decision.skill_confidence >= 0.70 else "moderate"

        router_guidance = (
            "## Router Tactical Guidance (Advisory)\n"
            f"- **Primary Intent Classified**: `{decision.skill}` (confidence tier: `{skill_tier}`, "
            f"complexity: `{decision.complexity}`)\n"
            f"- **Router Suggested Starting Tools**: `{recommended_tools}`\n"
            f"- **Active Sub-Skills Selected by Router**:\n{sub_skills_rendered}\n"
            "- **Autonomy Rule**: Treat the Router's suggestions above as an advisory guide, NOT a rigid cage. "
            "If conversation memory (`[Tool Memory]`) already contains the answer, answer directly without tools. "
            "Conversely, if a multi-step task requires chaining another available runtime tool "
            "(e.g., `file_search` -> `dir_tree` -> `file_read`), invoke it freely."
        )
        sections.append(router_guidance)

        return "\n\n".join(sections), " + ".join(loaded_labels)

    def _build_structured_conversation(
        self,
        system_prompt: str,
        history: list[dict[str, Any]] | None,
        user_message: str,
        needs_previous_context: bool = True,
    ) -> tuple[list[dict[str, str]], str, dict[str, int]]:
        """
        Builds the native conversation messages list (system / user / assistant) for Gemma 4.
        Uses English meta-thoughts in historical turns to prevent language drift, and compacts
        unrelated older tool outputs when needs_previous_context is False to speed up Prefill.
        """
        stats = {"full_memories": 0, "compacted_memories": 0}
        messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
        debug_lines: list[str] = []

        if history:
            recent_slice = history[-8:]
            last_assistant_idx = max(
                (i for i, t in enumerate(recent_slice) if t.get("role") == "assistant"),
                default=-1,
            )

            for idx, turn in enumerate(recent_slice):
                role = "assistant" if turn.get("role") == "assistant" else "user"
                clean_chat = str(turn.get("content", "")).strip()
                tool_memories: list[dict[str, Any]] = turn.get("tool_memories") or []

                if role == "assistant" and tool_memories:
                    is_latest_asst = (idx == last_assistant_idx)
                    mem_blocks: list[str] = []
                    for tm in tool_memories:
                        t_name = tm.get("tool", "tool")
                        t_out = self._strip_vision_markers(str(tm.get("output", "")).strip())
                        if is_latest_asst and needs_previous_context:
                            max_keep = 16000
                        elif is_latest_asst:
                            max_keep = 1800
                        else:
                            max_keep = 900

                        if len(t_out) <= max_keep:
                            mem_blocks.append(f"[Tool Memory: {t_name}]\n{t_out}\n[/Tool Memory]")
                            stats["full_memories"] += 1
                        else:
                            clipped = t_out[:max_keep] + f"\n... [Compacted to {max_keep} chars to optimize M4 KV memory]"
                            mem_blocks.append(f"[Tool Memory: {t_name}]\n{clipped}\n[/Tool Memory]")
                            stats["compacted_memories"] += 1

                    mem_str = "\n".join(mem_blocks)
                    combined_asst = (
                        "<thought>I executed the required tools and recorded their outputs in session memory.</thought>\n"
                        f"{mem_str}\n\n{clean_chat}"
                    )
                    messages.append({"role": "assistant", "content": combined_asst})
                    debug_lines.append(f"{mem_str}\nAssistant: {clean_chat}")
                elif role == "assistant":
                    combined_asst = (
                        "<thought>I answered the user's request based on the session context.</thought>\n"
                        f"{clean_chat}"
                    )
                    messages.append({"role": "assistant", "content": combined_asst})
                    debug_lines.append(f"Assistant: {clean_chat}")
                else:
                    messages.append({"role": role, "content": clean_chat})
                    debug_lines.append(f"{role.capitalize()}: {clean_chat}")

        messages.append({"role": "user", "content": user_message})
        debug_history_str = "\n".join(debug_lines)
        return messages, debug_history_str, stats

    def _format_tools_description(self, active_tool_names: set[str]) -> str:
        """Formats active tool schemas and single-action turn rules for the system prompt."""
        active_tools = {k: v for k, v in self.tool_registry.items() if k in active_tool_names}
        if not active_tools:
            return (
                "No external tools are active for this request.\n"
                "1. Start on line 1 with ONE <thought>...</thought> block (in the user's language, without backticks).\n"
                "2. Immediately after </thought>, write your complete answer in natural language."
            )

        lines = ["Available runtime tools:"]
        for name, meta in active_tools.items():
            params = json.dumps(meta["parameters"], ensure_ascii=False)
            lines.append(f"- {name}: {meta['description']}\n  Parameters JSON Schema: {params}")

        lines.append(
            "\nSingle-Action Turn Contract:\n"
            "In each step, you must perform ONLY ONE of the following two options:\n"
            "- OPTION A (Call ONE tool): Start with a single <thought>...</thought> block (never use parentheses like (thought), never emit two thought blocks), and on the next line output ONLY the raw JSON object:\n"
            '  {"action": "<tool_name>", "arguments": {"<param>": "<value>"}}\n'
            "- OPTION B (Final Answer to User): If you already have the needed data in the conversation or tool output, "
            "start with a single <thought>...</thought> block and write your complete natural language response in the exact language requested by the user (with ZERO JSON)."
        )
        return "\n".join(lines)

    @staticmethod
    def _normalize_gemma4_tokens(text: str) -> str:
        """
        Normalizes native Gemma 4 (<|channel>thought...<channel|>), bracketed ([thought]),
        and parenthesized ((thought)) thought markers into standard <thought>...</thought> tags.
        """
        g4_th_open = "<|channel>" + "thought"
        g4_th_close = "<channel" + "|>"
        th_open = "<" + "thought>"
        th_close = "</" + "thought>"
        empty_g4 = re.compile(rf"^\s*{re.escape(g4_th_open)}\s*{re.escape(g4_th_close)}\s*")
        cleaned = empty_g4.sub("", text)
        # Collapse any accidentally duplicated <|channel> prefixes before replacing
        cleaned = re.sub(r"(?:<\|channel>\s*)+thought", th_open, cleaned)
        cleaned = cleaned.replace(g4_th_open, th_open).replace(g4_th_close, th_close)
        cleaned = re.sub(r"^\s*\[[Tt]hought\]\s*", th_open, cleaned)
        cleaned = re.sub(r"\[/[Tt]hought\]", th_close, cleaned)
        cleaned = re.sub(r"^\s*\([Tt]hought\)\s*", th_open, cleaned)
        cleaned = re.sub(r"\(/[Tt]hought\)", th_close, cleaned)
        return cleaned

    @classmethod
    def _split_leading_thought(cls, raw_text: str) -> tuple[str, str]:
        """Splits one or more consecutive leading thought blocks from payload, handling unclosed streams gracefully."""
        text = cls._normalize_gemma4_tokens(raw_text)
        for tok in ("<" + "end_of_turn>", "<" + "start_of_turn>", "<" + "eos>", "<" + "turn|>", "<|turn>model"):
            text = text.replace(tok, "")
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
        # Strip any孤立/stray leading <|channel> token that was not followed by 'thought'
        if text.startswith("<|channel>"):
            text = text[len("<|channel>") :].lstrip()

        th_open = "<" + "thought>"
        th_close = "</" + "thought>"
        pattern = re.compile(rf"^[\s`]*{th_open}(.*?){th_close}[\s`]*(.*)", flags=re.DOTALL)

        thoughts: list[str] = []
        remaining = text
        while True:
            if remaining.startswith("<|channel>"):
                remaining = remaining[len("<|channel>") :].lstrip()
            match = pattern.match(remaining)
            if not match:
                break
            t_part = match.group(1).strip()
            if t_part:
                thoughts.append(t_part)
            remaining = match.group(2).strip()

        unclosed_pattern = rf"^[\s`]*{th_open}(.*)"
        unclosed_match = re.match(unclosed_pattern, remaining, flags=re.DOTALL)
        if unclosed_match:
            extra_t = unclosed_match.group(1).strip()
            if extra_t:
                thoughts.append(extra_t)
            return " ".join(thoughts), ""

        if thoughts:
            return " ".join(thoughts), remaining

        return "", remaining

    def _extract_tool_call(self, raw_text: str, raw_with_special: str = "") -> dict[str, Any] | None:
        """
        Parses a tool invocation from model output, supporting both standard JSON
        contracts ({"action": ...}) and native Gemma 4 tool calls (<|tool_call>call: name{args} or call: name{args}).
        """
        candidates = [raw_text]
        if raw_with_special and ("<|tool_call>" in raw_with_special or "call:" in raw_with_special or '"action"' in raw_with_special):
            candidates.insert(0, raw_with_special)

        for candidate in candidates:
            _, body = self._split_leading_thought(candidate)
            cleaned = body.strip()

            native_call_match = re.search(
                r"(?:<\|tool_call>)?call:\s*([a-zA-Z0-9_]+)\s*(\{.*?\})(?:<tool_call\|>|$)",
                cleaned,
                flags=re.DOTALL,
            )
            if native_call_match:
                action_name = native_call_match.group(1).strip()
                args_str = native_call_match.group(2).strip()
                args_norm = args_str.replace('<|"|>', '"')
                args_norm = re.sub(r'([{\s,])([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\1"\2":', args_norm)
                try:
                    args_dict = json.loads(args_norm)
                    if isinstance(args_dict, dict):
                        return {"action": action_name, "arguments": args_dict}
                except Exception:
                    try:
                        decoder = json.JSONDecoder()
                        args_dict, _ = decoder.raw_decode(args_norm)
                        if isinstance(args_dict, dict):
                            return {"action": action_name, "arguments": args_dict}
                    except Exception:
                        pass

            if cleaned.startswith("`"):
                cleaned = re.sub(r"^`+(?:json)?\s*", "", cleaned)
                cleaned = re.sub(r"\s*`+$", "", cleaned).strip()

            json_start_idx = cleaned.find("{")
            if json_start_idx != -1 and '"action"' in cleaned[json_start_idx:]:
                prefix_before_brace = cleaned[:json_start_idx].strip()
                if not prefix_before_brace or prefix_before_brace.lower().startswith("(thought)"):
                    repaired = re.sub(r',\s*"[^"]+"\s*:\s*$', "", cleaned[json_start_idx:])
                    open_b = repaired.count("{")
                    close_b = repaired.count("}")
                    if open_b > close_b:
                        repaired += "}" * (open_b - close_b)
                    end_idx = repaired.rfind("}")
                    if end_idx != -1:
                        json_candidate = repaired[: end_idx + 1]
                        try:
                            payload = json.loads(json_candidate)
                            if isinstance(payload, dict) and "action" in payload:
                                return payload
                        except json.JSONDecodeError:
                            try:
                                decoder = json.JSONDecoder()
                                obj, _ = decoder.raw_decode(repaired)
                                if isinstance(obj, dict) and "action" in obj:
                                    return obj
                            except Exception:
                                pass
        return None

    def _extract_final_answer(self, raw_text: str) -> str:
        """Extracts clean final text response, stripping accidental trailing tool call attempts or stray tags."""
        lead_thought, body = self._split_leading_thought(raw_text)
        body = re.sub(r"^Assistant:\s*", "", body).strip()
        body = re.sub(r"^\([Tt]hought\)\s*", "", body).strip()
        body = re.sub(r'\n+\s*\{\s*"action"\s*:.*$', "", body, flags=re.DOTALL).strip()
        if "</thought>" in body:
            body = body.split("</thought>", 1)[0].strip()

        if not body and lead_thought:
            return lead_thought

        if lead_thought and (
            body.startswith("Line ")
            or body.startswith("Línea ")
            or body.startswith("---")
            or body.startswith("def ")
        ):
            return f"{lead_thought}\n\n{body}"

        return body

    @staticmethod
    def _verify_and_repair_code_blocks(
        final_response: str,
        executed_tools_trace: list[dict[str, Any]],
    ) -> tuple[str, bool]:
        """
        Dynamic Ground-Truth AST Code Verifier:
        Verifies that the code block and line numbers displayed in final_response match the AST ground truth.
        """
        grep_traces = [t for t in executed_tools_trace if t.get("tool") == "file_grep"]
        if not grep_traces or "----------------------------------------" not in final_response:
            return final_response, False

        last_grep_out = str(grep_traces[-1].get("output", ""))
        header_match = re.search(
            r"(?:File|Archivo):\s*([^\n|]+)\s*\|\s*(?:Exact match on LINE|Coincidencia exacta en LÍNEA)\s*(\d+)\s*\[(?:COMPLETE DEFINITION: lines|DEFINICIÓN COMPLETA: líneas)\s*(\d+)\s*(?:to|a)\s*(\d+)\]:",
            last_grep_out,
        )
        if not header_match:
            return final_response, False

        start_line = header_match.group(3).strip()
        end_line = header_match.group(4).strip()

        ast_code_lines = [
            line for line in last_grep_out.splitlines()
            if re.match(r"^(?:Line|Línea)\s+\d+\s*\|", line)
        ]
        if not ast_code_lines:
            return final_response, False

        exact_ast_block = "\n".join(ast_code_lines)
        first_ast_line = ast_code_lines[0].strip()
        last_ast_line = ast_code_lines[-1].strip()

        if first_ast_line in final_response and last_ast_line in final_response:
            return final_response, False

        parts = final_response.split("----------------------------------------")
        if len(parts) >= 3:
            before_block = parts[0]
            after_block = "----------------------------------------".join(parts[2:])
            before_block = re.sub(
                r"(\*\*(?:Exact Line\(s\)|Exact Line of Definition|Línea exacta de definición)\*\*:\s*).*",
                rf"\g<1>Line {start_line} (complete definition: lines {start_line} to {end_line})",
                before_block,
            )
            repaired = (
                f"{before_block.rstrip()}\n"
                f"----------------------------------------\n"
                f"{exact_ast_block}\n"
                f"----------------------------------------"
                f"{after_block}"
            )
            return repaired, True

        return final_response, False

    def _should_use_apc_for_step(
        self,
        step: int,
        last_tool_name: str | None,
        has_active_vision: bool,
        user_message: str,
        decision: RoutingDecision,
    ) -> tuple[bool, str]:
        """
        Smart Hybrid APC Policy:
        - Vision steps: APC bypassed in favor of VisionFeatureCache.
        - Steps immediately following `dir_tree`, `file_read`, or `file_grep`:
          Bypass APC (`use_apc=False`) so Gemma 4's KV-sharing layers attend over exact filenames
          (e.g. `pato.png`), 7-digit identifiers (`1438516`), and AST lines (`321 to 382`) with 100% accuracy.
        - Multi-turn memory synthesis (`needs_previous_context=True` or explicit no-tool summary):
          Bypass APC so historical numbers/codes are synthesized with 100% fidelity.
        - Navigation & Tool-Dispatch steps (Step 1 initial tool dispatch, or Step 2 after `file_search`):
          Keep APC ENABLED (`use_apc=True`) for ultra-fast 1.1s–4.9s Prefill!
        """
        if has_active_vision:
            return False, "Vision step (using VisionFeatureCache)"
        if last_tool_name in {"dir_tree", "file_read", "file_grep"}:
            return False, f"High-precision synthesis after {last_tool_name} (ensures 100% exact filenames, digits & AST lines)"
        if step == 1 and (
            decision.skill == "personal_assistant"
            or decision.needs_previous_context
            or "without calling any tools" in user_message.lower()
        ):
            return False, "High-precision multi-turn session synthesis"
        if step == 1:
            return True, "Step 1 fast tool-dispatch (reusing Turn-0 static system prefix)"
        return True, f"Fast chained tool-dispatch after {last_tool_name}"

    def _count_tokens(self, text: str) -> int:
        tokenizer = getattr(self.llm, "tokenizer", None) or getattr(getattr(self.llm, "processor", None), "tokenizer", None)
        if tokenizer and hasattr(tokenizer, "encode"):
            try:
                return len(tokenizer.encode(text))
            except Exception:
                pass
        return max(1, len(text) // 4)

    def _generate_step(
        self,
        messages: list[dict[str, str]],
        images: list[str] | None = None,
        use_apc: bool = True,
        apc_reason: str = "standard",
        step_label: str = "STEP",
        on_token: Callable[[str], None] | None = None,
        on_thought_token: Callable[[str], None] | None = None,
        on_thought_done: Callable[[float], None] | None = None,
    ) -> dict[str, Any]:
        stop_literals = {"<" + "end_of_turn>", "<" + "start_of_turn>", "<" + "eos>", "<" + "turn|>"}
        th_open = "<" + "thought>"
        th_close = "</" + "thought>"
        fence_prefix = "`" * 3

        t_start = time.perf_counter()
        t_first_token: float | None = None
        buffer: list[str] = []
        streamed_answer = False
        thought_closed_notified = False
        captured_thought: list[str] = []

        if hasattr(self.llm, "stream"):
            mode = "DETECT"  # DETECT | THOUGHT | AFTER_THOUGHT | TOOL_JSON | ANSWER
            pending = ""

            for piece in self.llm.stream(
                messages,
                images=images,
                max_tokens=2048,
                use_apc=use_apc,
                apc_reason=apc_reason,
                step_label=step_label,
            ):
                if t_first_token is None:
                    t_first_token = time.perf_counter()

                if piece in stop_literals or any(s in piece for s in stop_literals):
                    clean_piece = piece
                    for s in stop_literals:
                        clean_piece = clean_piece.replace(s, "")
                    if clean_piece:
                        buffer.append(clean_piece)
                        if mode == "ANSWER" and on_token:
                            on_token(clean_piece.replace("\r", "\n"))
                        elif mode == "THOUGHT" and on_thought_token:
                            on_thought_token(clean_piece.replace("\r", "\n"))
                    break

                if mode == "ANSWER" and thought_closed_notified and "<channel|>" in piece:
                    before_ch = piece.split("<channel|>", 1)[0]
                    if before_ch:
                        buffer.append(before_ch)
                        if on_token:
                            on_token(before_ch.replace("\r", "\n"))
                    break

                buffer.append(piece)
                pending = self._normalize_gemma4_tokens(pending + piece)

                if mode == "DETECT":
                    stripped = pending.lstrip(" \t\r\n`")
                    if not stripped:
                        continue
                    if stripped.startswith(th_open):
                        mode = "THOUGHT"
                        pending = stripped[len(th_open) :].lstrip("\n ")
                    elif (
                        stripped.startswith("<")
                        or stripped.startswith("[")
                        or stripped.startswith("(")
                        or "call:".startswith(stripped)
                    ) and (
                        th_open.startswith(stripped)
                        or ("<|channel>" + "thought").startswith(stripped)
                        or ("<|channel><thought>").startswith(stripped)
                        or ("<|tool_call>").startswith(stripped)
                        or "call:".startswith(stripped)
                        or "[Thought]".startswith(stripped)
                        or "[thought]".startswith(stripped)
                        or "(Thought)".startswith(stripped)
                        or "(thought)".startswith(stripped)
                    ):
                        continue
                    elif (
                        stripped.startswith("{")
                        or stripped.startswith(fence_prefix)
                        or stripped.startswith("<|tool_call>")
                        or stripped.startswith("call:")
                    ):
                        mode = "TOOL_JSON"
                        pending = ""
                    elif len(stripped) >= 14 or "\n" in stripped:
                        mode = "ANSWER"
                        streamed_answer = True
                        if on_token:
                            on_token(stripped.replace("\r", "\n"))
                        pending = ""

                if mode == "THOUGHT":
                    if th_close in pending:
                        before_close, after_close = pending.split(th_close, 1)
                        if before_close:
                            captured_thought.append(before_close)
                            if on_thought_token:
                                on_thought_token(before_close.replace("\r", "\n"))
                        mode = "AFTER_THOUGHT"
                        pending = after_close.lstrip("`")
                    else:
                        tag_pos = pending.find("<")
                        bracket_pos = pending.find("[/")
                        paren_pos = pending.find("(/")
                        candidates = [p for p in (tag_pos, bracket_pos, paren_pos) if p != -1]
                        cut_pos = min(candidates) if candidates else -1

                        if cut_pos == -1:
                            if pending:
                                captured_thought.append(pending)
                                if on_thought_token:
                                    on_thought_token(pending.replace("\r", "\n"))
                            pending = ""
                        elif cut_pos > 0:
                            to_emit = pending[:cut_pos]
                            captured_thought.append(to_emit)
                            if on_thought_token:
                                on_thought_token(to_emit.replace("\r", "\n"))
                            pending = pending[cut_pos:]
                        elif len(pending) > len(th_close):
                            captured_thought.append(pending)
                            if on_thought_token:
                                on_thought_token(pending.replace("\r", "\n"))
                            pending = ""

                if mode == "AFTER_THOUGHT":
                    stripped_after = pending.lstrip(" \t\r\n`")
                    if not stripped_after:
                        continue
                    if stripped_after.startswith(th_open):
                        mode = "THOUGHT"
                        if captured_thought and on_thought_token:
                            on_thought_token(" ")
                        pending = stripped_after[len(th_open) :].lstrip("\n ")
                        continue
                    if (
                        stripped_after.startswith("<")
                        or stripped_after.startswith("[")
                        or stripped_after.startswith("(")
                        or "call:".startswith(stripped_after)
                    ) and (
                        th_open.startswith(stripped_after)
                        or ("<|channel>" + "thought").startswith(stripped_after)
                        or ("<|channel><thought>").startswith(stripped_after)
                        or ("<|tool_call>").startswith(stripped_after)
                        or "call:".startswith(stripped_after)
                        or "[Thought]".startswith(stripped_after)
                        or "[thought]".startswith(stripped_after)
                        or "(Thought)".startswith(stripped_after)
                        or "(thought)".startswith(stripped_after)
                    ):
                        continue
                    if (
                        stripped_after.startswith("{")
                        or stripped_after.startswith(fence_prefix)
                        or stripped_after.startswith("<|tool_call>")
                        or stripped_after.startswith("call:")
                    ):
                        if on_thought_done and not thought_closed_notified:
                            on_thought_done(time.perf_counter() - t_start)
                            thought_closed_notified = True
                        mode = "TOOL_JSON"
                        pending = ""
                    elif len(stripped_after) >= 14 or "\n" in stripped_after:
                        if on_thought_done and not thought_closed_notified:
                            on_thought_done(time.perf_counter() - t_start)
                            thought_closed_notified = True
                        mode = "ANSWER"
                        streamed_answer = True
                        if on_token:
                            thought_summary = "".join(captured_thought).strip()
                            if (
                                stripped_after.startswith("Line ")
                                or stripped_after.startswith("Línea ")
                                or stripped_after.startswith("---")
                                or stripped_after.startswith("def ")
                            ) and thought_summary:
                                on_token(f"{thought_summary}\n\n")
                            on_token(stripped_after.replace("\r", "\n"))
                        pending = ""

                elif mode == "TOOL_JSON":
                    pending = ""

                elif mode == "ANSWER":
                    if re.search(r'\n\s*\{\s*"action"', pending):
                        before_json = re.split(r'\n\s*\{\s*"action"', pending, maxsplit=1)[0]
                        if before_json and on_token:
                            on_token(before_json.replace("\r", "\n"))
                        pending = ""
                        break
                    if re.search(r'\n\s*(?:\{\s*(?:"(?:a(?:c(?:t(?:i(?:o(?:n)?)?)?)?)?)?)?)?$', pending):
                        continue
                    if pending and on_token:
                        streamed_answer = True
                        on_token(pending.replace("\r", "\n"))
                        pending = ""

            if (mode in {"THOUGHT", "AFTER_THOUGHT"} or captured_thought) and on_thought_done and not thought_closed_notified:
                on_thought_done(time.perf_counter() - t_start)
                thought_closed_notified = True

            if mode in {"DETECT", "AFTER_THOUGHT"} and pending.strip():
                rem = pending.strip().lstrip("`")
                if not (rem.startswith("{") or rem.startswith(fence_prefix) or rem.startswith("call:") or rem.startswith("<|tool_call>")):
                    streamed_answer = True
                    if on_token:
                        on_token(rem.replace("\r", "\n"))
            elif mode == "ANSWER" and pending:
                clean_tail = re.sub(r'\n+\s*\{\s*"action".*$', "", pending, flags=re.DOTALL).rstrip()
                if clean_tail and on_token:
                    streamed_answer = True
                    on_token(clean_tail.replace("\r", "\n"))

            t_end = time.perf_counter()
            ttft = (t_first_token - t_start) if t_first_token else (t_end - t_start)
            decode_sec = max(0.001, (t_end - t_first_token)) if t_first_token else 0.001
            full_out = "".join(buffer)
            return {
                "text": full_out,
                "streamed": streamed_answer,
                "final_mode": mode,
                "ttft_sec": ttft,
                "decode_sec": decode_sec,
                "total_sec": t_end - t_start,
                "llm_diagnostics": dict(getattr(self.llm, "last_diagnostics", {})),
            }

        full_text = self.llm.generate(
            messages,
            images=images,
            use_apc=use_apc,
            apc_reason=apc_reason,
            step_label=step_label,
        )
        t_end = time.perf_counter()
        return {
            "text": full_text,
            "streamed": False,
            "final_mode": "GENERATE_NO_STREAM",
            "ttft_sec": t_end - t_start,
            "decode_sec": 0.001,
            "total_sec": t_end - t_start,
            "llm_diagnostics": dict(getattr(self.llm, "last_diagnostics", {})),
        }

    def _describe_workflow_activity(
        self,
        step: int,
        last_tool: str | None,
        last_args: dict[str, Any] | None,
        skill: str,
        has_active_vision: bool = False,
    ) -> str:
        if has_active_vision:
            fp = Path(str((last_args or {}).get("filepath", "image"))).name
            return f"Visually inspecting pixels and metadata of «{fp}» with Gemma 4 Vision..."
        if step == 1:
            if skill == "file_management":
                return "Evaluating session history and target files/directories to inspect..."
            if skill == "coding":
                return "Preparing AST and codebase search..."
            return "Analyzing conversation context to formulate response..."

        if last_tool == "file_search":
            pat = (last_args or {}).get("pattern", "item")
            return f"Analyzing discovered file paths for «{pat}»..."
        if last_tool == "dir_tree":
            dp = Path(str((last_args or {}).get("dirpath", "directory"))).name
            return f"Selecting target file inside «{dp}»..."
        if last_tool == "file_read":
            fp = Path(str((last_args or {}).get("filepath", "document"))).name
            return f"Synthesizing extracted content from «{fp}»..."
        if last_tool == "file_grep":
            q = (last_args or {}).get("query", "code")
            return f"Preparing code block and technical explanation for «{q}»..."
        return "Synthesizing findings to write final response..."

    def _describe_tool_execution(self, action_name: str, args: dict[str, Any]) -> str:
        if action_name == "file_search":
            pat = args.get("pattern", "*")
            etype = args.get("entry_type", "all")
            return f"Searching for «{pat}» (type: {etype}) across authorized workspaces..."
        if action_name == "dir_tree":
            dp = args.get("dirpath", "")
            name = Path(str(dp)).name or dp
            ext = args.get("extension_filter", "")
            ext_info = f" [filter: {ext}]" if ext else ""
            return f"Exploring directory tree of «{name}»{ext_info}..."
        if action_name == "file_read":
            fp = args.get("filepath", "")
            name = Path(str(fp)).name or fp
            return f"Reading and inspecting «{name}»..."
        if action_name == "file_grep":
            q = args.get("query", "")
            ext = args.get("file_extension", "")
            ext_info = f" (files: {ext})" if ext else ""
            return f"Extracting complete AST definition of «{q}»{ext_info}..."
        return f"Executing tool {action_name}..."

    def _summarize_tool_result(self, action_name: str, result: str) -> str:
        first_line = result.strip().splitlines()[0] if result.strip() else ""
        if result.startswith("Error"):
            return f"⚠ {first_line}"
        if action_name == "file_search":
            if "Found items" in first_line or "encontrados" in first_line:
                paths = [l.strip("- ").strip() for l in result.splitlines()[1:] if l.strip().startswith("-")]
                first_item = paths[0] if paths else ""
                return f"✓ {first_line} {first_item}"
            return f"ℹ {first_line[:80]}"
        if action_name == "dir_tree":
            entries = len([l for l in result.splitlines() if "📄" in l or "📁" in l])
            return f"✓ Directory tree listed ({entries} visible entries)"
        if action_name == "file_read":
            vision_paths = self._extract_vision_paths(result)
            if vision_paths:
                img_name = Path(vision_paths[0]).name
                return f"🖼️ Image «{img_name}» + technical metadata loaded into Computer Vision pipeline"
            chars = len(result)
            pages = len(re.findall(r"--- (?:Page|Página) \d+", result))
            page_info = f"{pages} pages, " if pages else ""
            return f"✓ Read complete ({page_info}{chars} chars extracted)"
        if action_name == "file_grep":
            return f"✓ {first_line}"
        return "✓ Completed"

    def _log_step_diagnostics(
        self,
        turn_num: int,
        step_num: int,
        messages: list[dict[str, str]],
        step_res: dict[str, Any],
        lead_thought: str,
        tool_call: dict[str, Any] | None,
    ) -> None:
        diag = step_res.get("llm_diagnostics", {})
        msg_summary = "\n".join(
            f"  [{i}] role={m['role']} ({len(m['content'])} chars): {repr(m['content'][:160])}..."
            if len(m["content"]) > 160
            else f"  [{i}] role={m['role']} ({len(m['content'])} chars): {repr(m['content'])}"
            for i, m in enumerate(messages)
        )
        formatted_prompt = diag.get("formatted_prompt", "(unavailable)")
        log_block = (
            f"\n{'-' * 88}\n"
            f"[TURN {turn_num} | STEP {step_num}] INFERENCE DIAGNOSTICS\n"
            f"{'-' * 88}\n"
            f"1. STRUCTURED MESSAGES SENT TO _format_input ({len(messages)} messages):\n{msg_summary}\n\n"
            f"2. TEMPLATE METHOD: {diag.get('template_method', 'unknown')}\n"
            f"   KV CACHE STATUS: {diag.get('kv_cache_status', 'n/a')}\n"
            f"   ACTIVE VISION IMAGES: count={diag.get('images_count', 0)} | sizes={diag.get('images_dimensions', [])}\n"
            f"3. EXACT FORMATTED PROMPT (LAST 2000 CHARS):\n"
            f"----------------------------------------\n"
            f"{formatted_prompt[-2000:]}\n"
            f"----------------------------------------\n\n"
            f"4. LLM STREAM METRICS & STOP REASON:\n"
            f"   - TTFT (Prefill): {step_res['ttft_sec']:.2f}s | Decode: {step_res['decode_sec']:.2f}s | Total: {step_res['total_sec']:.2f}s\n"
            f"   - Final Stream State Machine Mode: {step_res.get('final_mode')}\n"
            f"   - Stop Reason: {diag.get('stop_reason')}\n"
            f"   - Total Chunks: {diag.get('total_chunks')} | Buffered Subword Chunks: {diag.get('buffered_subword_chunks')}\n"
            f"   - Buffered Subwords Sample (skip_special_tokens=False): {diag.get('buffered_tokens_sample')}\n\n"
            f"5. GENERATED TOKEN ID TRACE (ID:'token_str'):\n"
            f"   {diag.get('token_id_trace')}\n\n"
            f"6. RAW OUTPUT WITH SPECIAL TOKENS VISIBLE (skip_special_tokens=False):\n"
            f"   {repr(diag.get('raw_with_special_tokens', ''))}\n\n"
            f"7. RAW MODEL TEXT OUTPUT (skip_special_tokens=True):\n"
            f"   {repr(step_res['text'])}\n\n"
            f"8. PARSED THOUGHT:\n   {repr(lead_thought)}\n\n"
            f"9. PARSED TOOL CALL:\n   {json.dumps(tool_call, ensure_ascii=False) if tool_call else 'None'}\n"
        )
        self._append_trace_log(log_block)

    def run(
        self,
        user_message: str,
        history: list[dict[str, Any]] | None = None,
        on_token: Callable[[str], None] | None = None,
        on_status: Callable[[str], None] | None = None,
        on_thought_token: Callable[[str], None] | None = None,
        on_thought_done: Callable[[float], None] | None = None,
        on_event: Callable[[str, str], None] | None = None,
    ) -> dict[str, Any]:
        t_start_total = time.perf_counter()
        self.turn_counter += 1
        turn_num = self.turn_counter
        profiling: dict[str, Any] = {"steps": []}

        if on_status:
            on_status("Classifying intent and modular sub-skills with Intern-Decision...")

        t_route_start = time.perf_counter()
        decision = self.decide_route(user_message, history=history)
        t_route_elapsed = time.perf_counter() - t_route_start
        profiling["route_time_sec"] = t_route_elapsed

        active_tools = self._resolve_active_tools(decision)
        skill_content, loaded_skill_label = self._compose_modular_skill(
            decision=decision,
            active_tools=active_tools,
        )
        profiling["loaded_skill"] = loaded_skill_label

        if on_event:
            reroute_tag = " [Re-routed 2x]" if decision.rerouted else ""
            on_event(
                "route",
                f"Router{reroute_tag} → [{decision.skill} ({decision.skill_confidence:.2f}) | {decision.complexity}] "
                f"({loaded_skill_label}) ({t_route_elapsed:.2f}s)",
            )

        base_prompt = self._load_base_prompt()
        core_profile = self._load_core_profile()
        tools_prompt = self._format_tools_description(active_tools)

        system_prompt = (
            f"{base_prompt}\n\n"
            f"{core_profile}\n\n"
            f"--- RUNTIME TOOLS ---\n"
            f"{tools_prompt}\n\n"
            f"--- ACTIVE SKILL: {decision.skill.upper()} ---\n"
            f"{skill_content}"
        )

        messages, debug_history_str, mem_stats = self._build_structured_conversation(
            system_prompt=system_prompt,
            history=history,
            user_message=user_message,
            needs_previous_context=decision.needs_previous_context,
        )
        profiling["injected_history"] = debug_history_str
        profiling["memory_stats"] = mem_stats

        self._append_trace_log(
            f"\n{'#' * 88}\n"
            f"TURN {turn_num} START — {datetime.now().strftime('%H:%M:%S')}\n"
            f"USER INPUT (repr): {repr(user_message)}\n"
            f"ROUTER: skill={decision.skill} (conf={decision.skill_confidence:.2f}, rerouted={decision.rerouted}, "
            f"needs_prev_ctx={decision.needs_previous_context}) | sub_skills={sorted(list(decision.sub_skills))} "
            f"| loaded=({loaded_skill_label}) | complexity={decision.complexity} "
            f"| route_time={t_route_elapsed:.2f}s | active_tools={sorted(list(active_tools))}\n"
            f"SYSTEM PROMPT SIZE: {len(system_prompt)} chars\n"
            f"PROBABILITIES: {decision.probabilities}\n"
            f"MEMORY STATS AT TURN START: {mem_stats}\n"
            f"{'#' * 88}"
        )

        executed_tools_trace: list[dict[str, Any]] = []
        executed_signatures: set[str] = set()
        active_step_images: list[str] = []
        final_response = ""
        was_streamed_to_terminal = False

        last_tool_name: str | None = None
        last_tool_args: dict[str, Any] | None = None

        request_reminder = (
            f'\n(CRITICAL: Start with <thought>...</thought> and write your response in the EXACT language of the user request: "{user_message}")'
            if len(user_message) <= 500
            else "\n(CRITICAL: Start with <thought>...</thought> and write your response in the EXACT language used by the user.)"
        )

        for step in range(1, self.max_steps + 1):
            if on_status:
                activity_msg = self._describe_workflow_activity(
                    step,
                    last_tool_name,
                    last_tool_args,
                    decision.skill,
                    has_active_vision=bool(active_step_images),
                )
                on_status(activity_msg)

            use_apc_step, apc_reason = self._should_use_apc_for_step(
                step=step,
                last_tool_name=last_tool_name,
                has_active_vision=bool(active_step_images),
                user_message=user_message,
                decision=decision,
            )

            step_res = self._generate_step(
                messages,
                images=active_step_images if active_step_images else None,
                use_apc=use_apc_step,
                apc_reason=apc_reason,
                step_label=f"TURN {turn_num} | STEP {step}",
                on_token=on_token,
                on_thought_token=on_thought_token,
                on_thought_done=on_thought_done,
            )
            raw_step_output = step_res["text"]
            step_streamed = step_res["streamed"]
            t_step_elapsed = step_res["total_sec"]
            ttft_sec = step_res["ttft_sec"]
            decode_sec = step_res["decode_sec"]

            kv_status_str = str(step_res.get("llm_diagnostics", {}).get("kv_cache_status", ""))
            kv_short = ""
            m_match = re.search(r"matched=(\d+)\s*tok", kv_status_str)
            if m_match:
                kv_short = f"APC matched={m_match.group(1)} tok"
            elif "vision_step" in kv_status_str:
                kv_short = "VisionCache active"
            elif "HIGH_PRECISION_CLEAN_PREFILL" in kv_status_str:
                kv_short = "Clean High-Precision Prefill"

            tokens_gen = self._count_tokens(raw_step_output)
            decode_tps = round(tokens_gen / decode_sec, 1) if decode_sec > 0 else 0.0

            lead_thought, body_after_thought = self._split_leading_thought(raw_step_output)
            raw_special = step_res.get("llm_diagnostics", {}).get("raw_with_special_tokens", "")
            tool_call = self._extract_tool_call(raw_step_output, raw_with_special=raw_special)

            self._log_step_diagnostics(
                turn_num=turn_num,
                step_num=step,
                messages=messages,
                step_res=step_res,
                lead_thought=lead_thought,
                tool_call=tool_call,
            )

            if tool_call:
                action_name = str(tool_call.get("action", "")).strip()
                args = tool_call.get("arguments", {}) if isinstance(tool_call.get("arguments"), dict) else {}
                assistant_step_msg = (
                    f"<thought>{lead_thought}</thought>\n"
                    f'{json.dumps({"action": action_name, "arguments": args}, ensure_ascii=False)}'
                )

                signature = f"{action_name}:{json.dumps(args, sort_keys=True)}"
                if signature in executed_signatures:
                    profiling["steps"].append({
                        "step": step,
                        "type": "circuit_breaker",
                        "action": action_name,
                    })
                    self._append_trace_log(f"[TURN {turn_num} | STEP {step}] CIRCUIT BREAKER: Repeated {signature}")
                    messages.append({"role": "assistant", "content": assistant_step_msg})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM RUNTIME NOTICE]: You already executed '{action_name}' with these exact parameters. "
                            f"Do NOT call any more tools. Write your complete final answer in natural language now.{request_reminder}"
                        ),
                    })
                    continue
                executed_signatures.add(signature)

                if action_name in active_tools and action_name in self.tool_registry:
                    if on_status:
                        on_status(self._describe_tool_execution(action_name, args))

                    t_tool_start = time.perf_counter()
                    tool_func = self.tool_registry[action_name]["function"]
                    tool_result = str(tool_func(**args))
                    t_tool_elapsed = time.perf_counter() - t_tool_start

                    detected_imgs = self._extract_vision_paths(tool_result)
                    if detected_imgs:
                        active_step_images = detected_imgs[-2:]
                    else:
                        active_step_images = []

                    if on_event:
                        summary_badge = self._summarize_tool_result(action_name, tool_result)
                        on_event("tool", f"[{action_name}] {summary_badge} ({t_tool_elapsed:.2f}s)")

                    last_tool_name = action_name
                    last_tool_args = args

                    clean_result_for_trace = self._strip_vision_markers(tool_result)
                    executed_tools_trace.append({
                        "tool": action_name,
                        "args": args,
                        "output": clean_result_for_trace,
                    })
                    profiling["steps"].append({
                        "step": step,
                        "type": "tool_call",
                        "action": action_name,
                        "time_sec": t_step_elapsed,
                        "ttft_sec": ttft_sec,
                        "decode_sec": decode_sec,
                        "tokens": tokens_gen,
                        "tps": decode_tps,
                        "tool_exec_sec": t_tool_elapsed,
                        "kv_short": kv_short,
                    })

                    self._append_trace_log(
                        f"[TURN {turn_num} | STEP {step}] TOOL EXECUTED: {action_name}({json.dumps(args, ensure_ascii=False)}) "
                        f"in {t_tool_elapsed:.3f}s | vision_images={active_step_images}\n"
                        f"TOOL OUTPUT ({len(tool_result)} chars):\n"
                        f"----------------------------------------\n{tool_result}\n----------------------------------------"
                    )

                    messages.append({"role": "assistant", "content": assistant_step_msg})
                    messages.append({
                        "role": "user",
                        "content": (
                            f'[SYSTEM RUNTIME — AUTOMATED TOOL OUTPUT FOR YOUR PREVIOUS ACTION "{action_name}"]\n'
                            "(This output was generated automatically by the runtime when executing your tool call — NOT sent by the human user.)\n"
                            f'<tool_output tool="{action_name}">\n{clean_result_for_trace}\n</tool_output>\n\n'
                            "Review the <tool_output> above (and the attached image in your visual input if present). "
                            "Choose ONLY ONE option:\n"
                            "- Option A (Need another tool, e.g. to inspect another candidate file/image): Start with <thought>...</thought> "
                            "(never use backticks or parentheses) followed on the next line by ONLY the raw JSON tool call.\n"
                            "- Option B (Have all required data): Start with <thought>...</thought> (never use backticks) "
                            "followed by your complete final answer in natural language, and STOP immediately without any JSON."
                            f"{request_reminder}"
                        ),
                    })
                    continue
                else:
                    profiling["steps"].append({
                        "step": step,
                        "type": "circuit_breaker",
                        "action": action_name or "invalid_tool",
                    })
                    messages.append({"role": "assistant", "content": assistant_step_msg})
                    messages.append({
                        "role": "user",
                        "content": (
                            f"[SYSTEM RUNTIME NOTICE]: Tool '{action_name}' is not available. "
                            f"Write your final response directly in natural language.{request_reminder}"
                        ),
                    })
                    continue

            if not body_after_thought.strip() and lead_thought and step < self.max_steps:
                self._append_trace_log(
                    f"[TURN {turn_num} | STEP {step}] WARNING: Stream ended inside <thought> without body/JSON. Requesting continuation."
                )
                messages.append({"role": "assistant", "content": f"<thought>{lead_thought}</thought>"})
                messages.append({
                    "role": "user",
                    "content": (
                        "[SYSTEM RUNTIME NOTICE]: Your previous response stopped right after <thought> without emitting "
                        "either the tool JSON or the final answer. Emit ONLY the tool JSON object (if you need a tool "
                        f"like dir_tree or file_read) or write your complete final answer in natural language now.{request_reminder}"
                    ),
                })
                continue

            final_response = self._extract_final_answer(raw_step_output)
            final_response, was_repaired = self._verify_and_repair_code_blocks(final_response, executed_tools_trace)
            if was_repaired:
                self._append_trace_log(
                    f"[TURN {turn_num} | STEP {step}] AST GROUND-TRUTH VERIFIER: Repaired code block lines to match exact AST extraction."
                )
                if on_event:
                    on_event("retry", "[AST Verifier] Verified & aligned exact line numbers from AST ground truth")

            was_streamed_to_terminal = step_streamed
            profiling["steps"].append({
                "step": step,
                "type": "final_answer",
                "time_sec": t_step_elapsed,
                "ttft_sec": ttft_sec,
                "decode_sec": decode_sec,
                "tokens": tokens_gen,
                "tps": decode_tps,
                "kv_short": kv_short,
            })
            break

        if not final_response.strip():
            if on_status:
                on_status("Composing final response...")
            messages.append({
                "role": "user",
                "content": f"Tool phase finished. Provide your complete natural language response now.{request_reminder}",
            })
            fb_res = self._generate_step(
                messages,
                images=active_step_images if active_step_images else None,
                use_apc=False,
                apc_reason="Fallback final answer synthesis",
                step_label=f"TURN {turn_num} | FALLBACK",
                on_token=on_token,
                on_thought_token=on_thought_token,
                on_thought_done=on_thought_done,
            )
            fb_thought, _ = self._split_leading_thought(fb_res["text"])
            self._log_step_diagnostics(
                turn_num=turn_num,
                step_num=len(profiling["steps"]) + 1,
                messages=messages,
                step_res=fb_res,
                lead_thought=fb_thought,
                tool_call=None,
            )
            final_response = self._extract_final_answer(fb_res["text"])
            final_response, _ = self._verify_and_repair_code_blocks(final_response, executed_tools_trace)
            was_streamed_to_terminal = fb_res["streamed"]
            profiling["steps"].append({
                "step": len(profiling["steps"]) + 1,
                "type": "final_answer",
                "time_sec": fb_res["total_sec"],
                "ttft_sec": fb_res["ttft_sec"],
                "decode_sec": fb_res["decode_sec"],
                "tokens": self._count_tokens(final_response),
                "tps": round(self._count_tokens(final_response) / fb_res["decode_sec"], 1),
                "kv_short": "Clean High-Precision Prefill",
            })

        if not final_response.strip():
            if executed_tools_trace:
                last_t = executed_tools_trace[-1]
                final_response = f"Executed `{last_t['tool']}`, returning:\n\n{last_t['output']}"
            else:
                final_response = "Sorry Juan, an unexpected issue occurred while generating the response."

        if on_status:
            on_status("")

        if not was_streamed_to_terminal and on_token:
            on_token(final_response)

        profiling["total_time_sec"] = time.perf_counter() - t_start_total
        profiling["turn_num"] = turn_num
        self._append_trace_log(
            f"\n[TURN {turn_num} COMPLETE] Total Time: {profiling['total_time_sec']:.2f}s\n"
            f"FINAL RESPONSE DELIVERED ({len(final_response)} chars):\n"
            f"----------------------------------------\n{final_response}\n----------------------------------------\n"
        )

        return {
            "response": final_response,
            "decision": decision,
            "tools_trace": executed_tools_trace,
            "active_tools": active_tools,
            "profiling": profiling,
        }