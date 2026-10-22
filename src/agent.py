"""Orquestador del agente basado en mensajes multi-turno nativos (messages), herramientas SRP y logs acumulativos por sesión."""
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
from router.schema import question_schema
from tools.file_tools import TOOL_METADATA

ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent
SKILLS_DIR = ROOT_DIR / "skills"
CONTEXT_DIR = ROOT_DIR / "context"
LOGS_DIR = PROJECT_ROOT / "logs"

FILE_TOOLKIT = {"file_search", "dir_tree", "file_grep", "file_read"}

SKILL_TOOL_BUNDLES: dict[str, set[str]] = {
    "file_management": FILE_TOOLKIT,
    "coding": FILE_TOOLKIT,
}


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

    def reset_session_logs(self) -> None:
        """
        Se ejecuta ÚNICAMENTE al iniciar una sesión nueva (python src/main.py) o con el comando 'clear'.
        Inicializa desde cero ambos archivos (session_trace.log y session_history.md).
        Durante la sesión, ambos archivos solo reciben adiciones (append) turno tras turno.
        """
        self.turn_counter = 0
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
                f"- **Lifecycle**: Este archivo se inicializa solo al arrancar la sesión y **acumula (append)** "
                f"cada turno secuencialmente sin borrar turnos anteriores.\n\n"
                f"---\n\n"
            )
            self.history_log_path.write_text(history_header, encoding="utf-8")
        except Exception:
            pass

    def _append_trace_log(self, block: str) -> None:
        """Añade un bloque detallado al archivo session_trace.log con vaciado inmediato a disco."""
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            with self.trace_log_path.open("a", encoding="utf-8") as f:
                f.write(block + "\n")
                f.flush()
        except Exception:
            pass

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
        """
        Añade (append, mode='a') el registro completo del Turno N en logs/session_history.md
        sin borrar jamás lo registrado en los turnos previos de la misma sesión.
        """
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            ts = datetime.now().strftime("%H:%M:%S")
            lines = [
                f"## Turno {turn_num} — [{ts}]",
                f"- **Mensajes totales acumulados en `history` al cerrar este turno**: {total_history_messages} "
                f"({total_history_messages // 2} turnos completos)",
                f"- **Memorias de turnos previos inyectadas al iniciar el Turno {turn_num}**: "
                f"Completas={mem_stats.get('full_memories', 0)} | Compactadas={mem_stats.get('compacted_memories', 0)}\n",
                f"### 1. Contexto de sesión previo que leyó el modelo al iniciar el Turno {turn_num}:",
                "----------------------------------------",
                injected_history_snapshot.strip() if injected_history_snapshot.strip() else "(Turno inicial — sin historial previo)",
                "----------------------------------------\n",
                f"### 2. Solicitud del Usuario (`Juan`) en el Turno {turn_num}:",
                "----------------------------------------",
                user_message.strip(),
                "----------------------------------------\n",
            ]

            if stored_memories:
                lines.append(f"### 3. Nuevos bloques `[Tool Memory]` guardados en el Turno {turn_num} ({len(stored_memories)}):")
                for idx, tm in enumerate(stored_memories, start=1):
                    t_name = tm.get("tool", "tool")
                    t_out = str(tm.get("output", "")).strip()
                    lines.append(f"**3.{idx} Herramienta `{t_name}` ({len(t_out)} chars):**")
                    lines.append("----------------------------------------")
                    lines.append(t_out)
                    lines.append("----------------------------------------")
                lines.append("")
            else:
                lines.append(
                    f"### 3. Nuevos bloques `[Tool Memory]` guardados en el Turno {turn_num}:\n"
                    "*(Ninguna herramienta ejecutada en este turno — el modelo respondió usando memoria previa)*\n"
                )

            lines.extend([
                f"### 4. Respuesta Final del `Asistente` guardada en memoria (Turno {turn_num}):",
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

    def decide_route(self, user_message: str) -> RoutingDecision:
        try:
            request = {
                "state": {"user": user_message},
                "questions": question_schema("en"),
            }
            raw_result = self.decider.predict(request)
            decision = RoutingDecision.from_answers(raw_result["answers"])

            target_skill = (
                "personal_assistant"
                if decision.skill in {"email_assistant", "study_assistant", "research"}
                else decision.skill
            )
            filtered_tools = frozenset(t for t in decision.tools if t in self.tool_registry)

            return RoutingDecision(
                skill=target_skill,
                complexity=decision.complexity,
                model_route=decision.model_route,
                tools=filtered_tools,
                needs_clarification=decision.needs_clarification,
                requires_confirmation=decision.requires_confirmation,
                probabilities=decision.probabilities,
                latency_ms=decision.latency_ms,
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
            )

    def _resolve_active_tools(self, decision: RoutingDecision) -> set[str]:
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

    def _load_single_skill(self, skill_name: str, has_file_tools: bool) -> tuple[str, str]:
        skill_file = SKILLS_DIR / f"{skill_name}.md"
        if skill_file.exists():
            return skill_file.read_text(encoding="utf-8"), f"{skill_name}.md"

        if has_file_tools and (SKILLS_DIR / "file_management.md").exists():
            return (SKILLS_DIR / "file_management.md").read_text(encoding="utf-8"), "file_management.md"

        if (SKILLS_DIR / "personal_assistant.md").exists():
            return (SKILLS_DIR / "personal_assistant.md").read_text(encoding="utf-8"), "personal_assistant.md"

        return (
            "You are a helpful, direct, and grounded personal AI assistant for Juan Fernando.",
            "builtin_default",
        )

    def _build_structured_conversation(
        self,
        system_prompt: str,
        history: list[dict[str, Any]] | None,
        user_message: str,
    ) -> tuple[list[dict[str, str]], str, dict[str, int]]:
        """
        Construye la lista nativa de mensajes (system / user / assistant) para el chat_template
        de Gemma 4, calibrando la memoria en 16.000 chars para el último turno y 3.500 chars
        para turnos anteriores (equilibrio óptimo de RAM y latencia TTFT en Apple Silicon M4 16GB).
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
                        t_out = str(tm.get("output", "")).strip()
                        max_keep = 16000 if is_latest_asst else 3500
                        if len(t_out) <= max_keep:
                            mem_blocks.append(f"[Tool Memory: {t_name}]\n{t_out}\n[/Tool Memory]")
                            stats["full_memories"] += 1
                        else:
                            clipped = t_out[:max_keep] + f"\n... [Truncado a {max_keep} chars para optimizar memoria en M4]"
                            mem_blocks.append(f"[Tool Memory: {t_name}]\n{clipped}\n[/Tool Memory]")
                            stats["compacted_memories"] += 1

                    mem_str = "\n".join(mem_blocks)
                    combined_asst = f"{mem_str}\n\n{clean_chat}"
                    messages.append({"role": "assistant", "content": combined_asst})
                    debug_lines.append(f"{mem_str}\nAssistant: {clean_chat}")
                else:
                    messages.append({"role": role, "content": clean_chat})
                    debug_lines.append(f"{role.capitalize()}: {clean_chat}")

        messages.append({"role": "user", "content": user_message})
        debug_history_str = "\n".join(debug_lines)
        return messages, debug_history_str, stats

    def _format_tools_description(self, active_tool_names: set[str]) -> str:
        active_tools = {k: v for k, v in self.tool_registry.items() if k in active_tool_names}
        if not active_tools:
            return (
                "No external tools are active for this request.\n"
                "1. Start on line 1 with <thought>...</thought> (in Spanish, without backticks).\n"
                "2. Immediately after </thought>, write your complete answer in natural language."
            )

        lines = ["Available runtime tools:"]
        for name, meta in active_tools.items():
            params = json.dumps(meta["parameters"], ensure_ascii=False)
            lines.append(f"- {name}: {meta['description']}\n  Parameters JSON Schema: {params}")

        lines.append(
            "\nSingle-Action Turn Contract:\n"
            "In each turn, you must perform ONLY ONE of the following two options:\n"
            "- OPTION A (Call ONE tool): Start with <thought>...</thought> (no backticks), and on the next line output ONLY the raw JSON object:\n"
            '  {"action": "<tool_name>", "arguments": {"<param>": "<value>"}}\n'
            "- OPTION B (Final Answer to User): If you already have the needed data in the conversation or tool output, "
            "start with <thought>...</thought> and write your complete natural language response to the user (with ZERO JSON)."
        )
        return "\n".join(lines)

    @staticmethod
    def _normalize_gemma4_tokens(text: str) -> str:
        """
        Normaliza los delimitadores nativos de pensamiento de Gemma 4 (<|channel>thought ... <channel|>)
        hacia <thought>...</thought> sin alterar contenido de archivos inspeccionados.
        """
        g4_th_open = "<|channel>" + "thought"
        g4_th_close = "<channel" + "|>"
        th_open = "<" + "thought>"
        th_close = "</" + "thought>"
        empty_g4 = re.compile(rf"^\s*{re.escape(g4_th_open)}\s*{re.escape(g4_th_close)}\s*")
        cleaned = empty_g4.sub("", text)
        return cleaned.replace(g4_th_open, th_open).replace(g4_th_close, th_close)

    @classmethod
    def _split_leading_thought(cls, raw_text: str) -> tuple[str, str]:
        """
        Separa únicamente el bloque <thought>...</thought> inicial (incluso si quedó truncado sin
        etiqueta de cierre </thought>), evitando que se imprima '<thought>...' en la respuesta final.
        """
        text = cls._normalize_gemma4_tokens(raw_text)
        for tok in ("<" + "end_of_turn>", "<" + "start_of_turn>", "<" + "eos>", "<" + "turn|>"):
            text = text.replace(tok, "")
        text = text.replace("\r\n", "\n").replace("\r", "\n").strip()

        th_open = "<" + "thought>"
        th_close = "</" + "thought>"
        pattern = rf"^[\s`]*{th_open}(.*?){th_close}[\s`]*(.*)"
        match = re.match(pattern, text, flags=re.DOTALL)
        if match:
            return match.group(1).strip(), match.group(2).strip()

        unclosed_pattern = rf"^[\s`]*{th_open}(.*)"
        unclosed_match = re.match(unclosed_pattern, text, flags=re.DOTALL)
        if unclosed_match:
            return unclosed_match.group(1).strip(), ""

        return "", text

    def _extract_tool_call(self, raw_text: str) -> dict[str, Any] | None:
        _, body = self._split_leading_thought(raw_text)
        cleaned = body.strip()

        if cleaned.startswith("`"):
            cleaned = re.sub(r"^`+(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*`+$", "", cleaned).strip()

        if not cleaned.startswith("{"):
            return None

        repaired = re.sub(r',\s*"[^"]+"\s*:\s*$', "", cleaned)
        open_b = repaired.count("{")
        close_b = repaired.count("}")
        if open_b > close_b and '"action"' in repaired:
            repaired += "}" * (open_b - close_b)

        end_idx = repaired.rfind("}")
        if end_idx == -1:
            return None

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
        lead_thought, body = self._split_leading_thought(raw_text)
        body = re.sub(r"^Assistant:\s*", "", body).strip()
        body = re.sub(r'\n+\s*\{\s*"action"\s*:.*$', "", body, flags=re.DOTALL).strip()

        if not body and lead_thought:
            return lead_thought

        if lead_thought and (body.startswith("Línea ") or body.startswith("---") or body.startswith("def ")):
            return f"{lead_thought}\n\n{body}"

        return body

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

            for piece in self.llm.stream(messages, max_tokens=2048):
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

                buffer.append(piece)
                pending = self._normalize_gemma4_tokens(pending + piece)

                if mode == "DETECT":
                    stripped = pending.lstrip(" \t\r\n`")
                    if not stripped:
                        continue
                    if stripped.startswith(th_open):
                        mode = "THOUGHT"
                        pending = stripped[len(th_open) :].lstrip("\n ")
                    elif stripped.startswith("<") and (
                        th_open.startswith(stripped) or ("<|channel>" + "thought").startswith(stripped)
                    ):
                        continue
                    elif stripped.startswith("{") or stripped.startswith(fence_prefix):
                        mode = "TOOL_JSON"
                        pending = ""
                    elif len(stripped) >= 8 or "\n" in stripped:
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
                        if on_thought_done and not thought_closed_notified:
                            on_thought_done(time.perf_counter() - t_start)
                            thought_closed_notified = True
                        mode = "AFTER_THOUGHT"
                        pending = after_close.lstrip("`")
                    else:
                        lt_pos = pending.find("<")
                        if lt_pos == -1:
                            if pending:
                                captured_thought.append(pending)
                                if on_thought_token:
                                    on_thought_token(pending.replace("\r", "\n"))
                            pending = ""
                        elif lt_pos > 0:
                            to_emit = pending[:lt_pos]
                            captured_thought.append(to_emit)
                            if on_thought_token:
                                on_thought_token(to_emit.replace("\r", "\n"))
                            pending = pending[lt_pos:]
                        elif len(pending) > len(th_close):
                            captured_thought.append(pending)
                            if on_thought_token:
                                on_thought_token(pending.replace("\r", "\n"))
                            pending = ""

                if mode == "AFTER_THOUGHT":
                    stripped_after = pending.lstrip(" \t\r\n`")
                    if not stripped_after:
                        continue
                    if stripped_after.startswith("{") or stripped_after.startswith(fence_prefix):
                        mode = "TOOL_JSON"
                        pending = ""
                    elif len(stripped_after) >= 10 or "\n" in stripped_after:
                        mode = "ANSWER"
                        streamed_answer = True
                        if on_token:
                            thought_summary = "".join(captured_thought).strip()
                            if (
                                stripped_after.startswith("Línea ")
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

            if mode == "THOUGHT" and on_thought_done and not thought_closed_notified:
                on_thought_done(time.perf_counter() - t_start)

            if mode in {"DETECT", "AFTER_THOUGHT"} and pending.strip():
                rem = pending.strip().lstrip("`")
                if not (rem.startswith("{") or rem.startswith(fence_prefix)):
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

        full_text = self.llm.generate(messages)
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
    ) -> str:
        if step == 1:
            if skill == "file_management":
                return "Evaluando historial y qué archivos o directorios inspeccionar..."
            if skill == "coding":
                return "Preparando búsqueda en el código fuente del proyecto..."
            return "Analizando el contexto de la conversación para responder..."

        if last_tool == "file_search":
            pat = (last_args or {}).get("pattern", "el elemento")
            return f"Analizando las rutas encontradas para «{pat}»..."
        if last_tool == "dir_tree":
            dp = Path(str((last_args or {}).get("dirpath", "directorio"))).name
            return f"Identificando el archivo objetivo dentro de «{dp}»..."
        if last_tool == "file_read":
            fp = Path(str((last_args or {}).get("filepath", "documento"))).name
            return f"Sintetizando la información extraída de «{fp}»..."
        if last_tool == "file_grep":
            q = (last_args or {}).get("query", "código")
            return f"Preparando explicación y bloque de código de «{q}»..."
        return "Integrando los hallazgos para redactar la respuesta..."

    def _describe_tool_execution(self, action_name: str, args: dict[str, Any]) -> str:
        if action_name == "file_search":
            pat = args.get("pattern", "*")
            etype = args.get("entry_type", "all")
            return f"Buscando «{pat}» (tipo: {etype}) en tus carpetas autorizadas..."
        if action_name == "dir_tree":
            dp = args.get("dirpath", "")
            name = Path(str(dp)).name or dp
            ext = args.get("extension_filter", "")
            ext_info = f" [filtro: {ext}]" if ext else ""
            return f"Explorando estructura del directorio «{name}»{ext_info}..."
        if action_name == "file_read":
            fp = args.get("filepath", "")
            name = Path(str(fp)).name or fp
            return f"Leyendo contenido del archivo «{name}»..."
        if action_name == "file_grep":
            q = args.get("query", "")
            ext = args.get("file_extension", "")
            ext_info = f" (archivos {ext})" if ext else ""
            return f"Rastreando la definición completa de «{q}»{ext_info}..."
        return f"Ejecutando operación {action_name}..."

    def _summarize_tool_result(self, action_name: str, result: str) -> str:
        first_line = result.strip().splitlines()[0] if result.strip() else ""
        if result.startswith("Error"):
            return f"⚠ {first_line}"
        if action_name == "file_search":
            if "encontrados" in first_line:
                paths = [l.strip("- ").strip() for l in result.splitlines()[1:] if l.strip().startswith("-")]
                first_item = paths[0] if paths else ""
                return f"✓ {first_line} {first_item}"
            return f"ℹ {first_line[:80]}"
        if action_name == "dir_tree":
            entries = len([l for l in result.splitlines() if "📄" in l or "📁" in l])
            return f"✓ Árbol de directorio listado ({entries} entradas visibles)"
        if action_name == "file_read":
            chars = len(result)
            pages = len(re.findall(r"--- Página \d+", result))
            page_info = f"{pages} pág., " if pages else ""
            return f"✓ Lectura completada ({page_info}{chars} caracteres extraídos)"
        if action_name == "file_grep":
            return f"✓ {first_line}"
        return "✓ Completado"

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
            on_status("Clasificando intención con Intern-Decision...")

        t_route_start = time.perf_counter()
        decision = self.decide_route(user_message)
        t_route_elapsed = time.perf_counter() - t_route_start
        profiling["route_time_sec"] = t_route_elapsed

        active_tools = self._resolve_active_tools(decision)
        skill_content, loaded_skill_file = self._load_single_skill(decision.skill, bool(active_tools))
        profiling["loaded_skill"] = loaded_skill_file

        if on_event:
            on_event(
                "route",
                f"Router → [{decision.skill} | {decision.complexity}] (Skill: {loaded_skill_file}) ({t_route_elapsed:.2f}s)",
            )

        base_prompt = self._load_base_prompt()
        core_profile = self._load_core_profile()
        tools_prompt = self._format_tools_description(active_tools)

        system_prompt = (
            f"{base_prompt}\n\n"
            f"{core_profile}\n\n"
            f"--- ACTIVE SKILL: {decision.skill.upper()} ({loaded_skill_file}) ---\n"
            f"{skill_content}\n\n"
            f"--- RUNTIME TOOLS ---\n"
            f"{tools_prompt}"
        )

        messages, debug_history_str, mem_stats = self._build_structured_conversation(
            system_prompt=system_prompt,
            history=history,
            user_message=user_message,
        )
        profiling["injected_history"] = debug_history_str
        profiling["memory_stats"] = mem_stats

        self._append_trace_log(
            f"\n{'#' * 88}\n"
            f"TURN {turn_num} START — {datetime.now().strftime('%H:%M:%S')}\n"
            f"USER INPUT (repr): {repr(user_message)}\n"
            f"ROUTER: skill={decision.skill} ({loaded_skill_file}) | complexity={decision.complexity} "
            f"| route_time={t_route_elapsed:.2f}s | active_tools={sorted(list(active_tools))}\n"
            f"PROBABILITIES: {decision.probabilities}\n"
            f"MEMORY STATS AT TURN START: {mem_stats}\n"
            f"{'#' * 88}"
        )

        executed_tools_trace: list[dict[str, Any]] = []
        executed_signatures: set[str] = set()
        final_response = ""
        was_streamed_to_terminal = False

        last_tool_name: str | None = None
        last_tool_args: dict[str, Any] | None = None

        request_reminder = (
            f'\n(Remember to address every part of the user request: "{user_message}")'
            if len(user_message) <= 500
            else ""
        )

        for step in range(1, self.max_steps + 1):
            if on_status:
                activity_msg = self._describe_workflow_activity(step, last_tool_name, last_tool_args, decision.skill)
                on_status(activity_msg)

            step_res = self._generate_step(
                messages,
                on_token=on_token,
                on_thought_token=on_thought_token,
                on_thought_done=on_thought_done,
            )
            raw_step_output = step_res["text"]
            step_streamed = step_res["streamed"]
            t_step_elapsed = step_res["total_sec"]
            ttft_sec = step_res["ttft_sec"]
            decode_sec = step_res["decode_sec"]

            tokens_gen = self._count_tokens(raw_step_output)
            decode_tps = round(tokens_gen / decode_sec, 1) if decode_sec > 0 else 0.0

            lead_thought, body_after_thought = self._split_leading_thought(raw_step_output)
            tool_call = self._extract_tool_call(raw_step_output)

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
                            f"[System Notice]: You already executed '{action_name}' with these exact parameters. "
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
                    tool_result = tool_func(**args)
                    t_tool_elapsed = time.perf_counter() - t_tool_start

                    if on_event:
                        summary_badge = self._summarize_tool_result(action_name, str(tool_result))
                        on_event("tool", f"[{action_name}] {summary_badge} ({t_tool_elapsed:.2f}s)")

                    last_tool_name = action_name
                    last_tool_args = args

                    executed_tools_trace.append({"tool": action_name, "args": args, "output": tool_result})
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
                    })

                    self._append_trace_log(
                        f"[TURN {turn_num} | STEP {step}] TOOL EXECUTED: {action_name}({json.dumps(args, ensure_ascii=False)}) "
                        f"in {t_tool_elapsed:.3f}s\nTOOL OUTPUT ({len(str(tool_result))} chars):\n"
                        f"----------------------------------------\n{tool_result}\n----------------------------------------"
                    )

                    messages.append({"role": "assistant", "content": assistant_step_msg})
                    messages.append({
                        "role": "user",
                        "content": (
                            f'<tool_output tool="{action_name}">\n{tool_result}\n</tool_output>\n\n'
                            "Review the <tool_output> above. Choose ONLY ONE option:\n"
                            "- Option A (Need another tool): Start with <thought>...</thought> (never use backticks) "
                            "followed on the next line by ONLY the raw JSON tool call.\n"
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
                            f"[System Notice]: Tool '{action_name}' is not available. "
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
                        "[System Notice]: Your previous response stopped right after <thought> without emitting "
                        "either the tool JSON or the final answer. Emit ONLY the tool JSON object (if you need a tool "
                        f"like file_read) or write your complete final answer in natural language now.{request_reminder}"
                    ),
                })
                continue

            final_response = self._extract_final_answer(raw_step_output)
            was_streamed_to_terminal = step_streamed
            profiling["steps"].append({
                "step": step,
                "type": "final_answer",
                "time_sec": t_step_elapsed,
                "ttft_sec": ttft_sec,
                "decode_sec": decode_sec,
                "tokens": tokens_gen,
                "tps": decode_tps,
            })
            break

        if not final_response.strip():
            if on_status:
                on_status("Redactando respuesta final...")
            messages.append({
                "role": "user",
                "content": f"Tool phase finished. Provide your complete natural language response now.{request_reminder}",
            })
            fb_res = self._generate_step(
                messages,
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
            was_streamed_to_terminal = fb_res["streamed"]
            profiling["steps"].append({
                "step": len(profiling["steps"]) + 1,
                "type": "final_answer",
                "time_sec": fb_res["total_sec"],
                "ttft_sec": ttft_sec,
                "decode_sec": fb_res["decode_sec"],
                "tokens": self._count_tokens(final_response),
                "tps": round(self._count_tokens(final_response) / fb_res["decode_sec"], 1),
            })

        if not final_response.strip():
            if executed_tools_trace:
                last_t = executed_tools_trace[-1]
                final_response = f"Ejecuté `{last_t['tool']}`, cuyo resultado fue:\n\n{last_t['output']}"
            else:
                final_response = "Lo siento Juan, ocurrió un problema al generar la respuesta."

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