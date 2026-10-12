"""Orquestador del agente personal con enrutamiento dinámico y ejecución segura de herramientas."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from intern_decision_mlx import Decider
from llm.base import LLM
from router.decision import RoutingDecision
from router.schema import question_schema
from tools.file_tools import TOOL_METADATA

SKILLS_DIR = Path(__file__).resolve().parent / "skills"
CONTEXT_DIR = Path(__file__).resolve().parent / "context"


class Agent:
    def __init__(self, llm: LLM, decider: Decider | None = None) -> None:
        self.llm = llm
        self.decider = decider if decider is not None else Decider()
        self.tool_registry = TOOL_METADATA

    def decide_route(self, user_message: str) -> RoutingDecision:
        """Clasifica la intención del usuario mediante Intern-Decision-0.8B MLX."""
        request = {
            "state": {"user": user_message},
            "questions": question_schema("en"),
        }
        raw_result = self.decider.predict(request)
        return RoutingDecision.from_answers(raw_result["answers"])

    def _load_core_profile(self) -> str:
        """Carga la identidad y contexto base del usuario."""
        profile_file = CONTEXT_DIR / "core_profile.md"
        if profile_file.exists():
            return profile_file.read_text(encoding="utf-8")
        return ""

    def _load_skill(self, skill_name: str) -> str:
        """Carga el archivo Markdown de la skill desde disco."""
        skill_file = SKILLS_DIR / f"{skill_name}.md"
        if skill_file.exists():
            return skill_file.read_text(encoding="utf-8")

        # Instrucción de respaldo si la skill solicitada aún no tiene archivo .md
        return (
            "You are a helpful personal assistant. "
            "Analyze the request carefully and assist the user accurately."
        )

    def _format_tools_description(self, active_tool_names: frozenset[str]) -> str:
        """Genera el catálogo en texto para las herramientas activas según el enrutador."""
        active_tools = {
            name: self.tool_registry[name]
            for name in active_tool_names
            if name in self.tool_registry
        }

        if not active_tools:
            return "No external tools are available for this request."

        lines = ["Available tools:"]
        for name, meta in active_tools.items():
            params = json.dumps(meta["parameters"], ensure_ascii=False)
            lines.append(f"- `{name}`: {meta['description']}\n  Parameters JSON Schema: {params}")

        lines.append(
            "\nInvocation format:\n"
            "If you need to execute a tool, respond ONLY with a raw JSON object and nothing else:\n"
            '{"action": "<tool_name>", "arguments": {"<param>": "<value>"}}\n'
            "If no tool is required, respond directly to the user."
        )
        return "\n".join(lines)

    def _extract_tool_call(self, text: str) -> dict[str, Any] | None:
        """Parsea la llamada a herramienta extrayendo el bloque JSON, incluso si viene en bloques markdown."""
        cleaned = text.strip()

        # Eliminar posibles bloques de código markdown tipo ```json ... ```
        if "```" in cleaned:
            match_fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned, re.DOTALL)
            if match_fence:
                cleaned = match_fence.group(1).strip()

        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not match:
            return None

        try:
            payload = json.loads(match.group(0))
            if isinstance(payload, dict) and "action" in payload:
                return payload
        except json.JSONDecodeError:
            return None
        return None

    def run(self, user_message: str) -> dict[str, Any]:
        """Ciclo completo: Routing -> Carga de Skill/Tool -> Inferencia Gemma -> Tool Call -> Respuesta."""
        # 1. Enrutamiento con el modelo de decisión
        decision = self.decide_route(user_message)

        # 2. Carga en caliente del Markdown de la Skill y metadatos de las Tools
        core_context = self._load_core_profile()
        skill_content = self._load_skill(decision.skill)
        tools_prompt = self._format_tools_description(decision.tools)

        # 3. Ensamblado del System Prompt
        system_prompt = (
            f"{core_context}\n\n"
            f"--- ACTIVE SKILL ---\n"
            f"{skill_content}\n\n"
            f"--- RUNTIME TOOLS ---\n"
            f"{tools_prompt}"
        )

        initial_prompt = f"{system_prompt}\n\nUser request: {user_message}\nAssistant:"
        first_response = self.llm.generate(initial_prompt)

        tool_call = self._extract_tool_call(first_response)
        tool_executed = None
        tool_output = None
        final_response = first_response

        # 4. Verificación y ejecución determinista de la herramienta
        if tool_call and tool_call.get("action") in decision.tools:
            action_name = tool_call["action"]
            arguments = tool_call.get("arguments", {})

            if action_name in self.tool_registry:
                tool_func = self.tool_registry[action_name]["function"]
                tool_output = tool_func(**arguments)
                tool_executed = action_name

                # 5. Respuesta final delimitando el contenido en etiquetas XML
                followup_prompt = (
                    f"{initial_prompt}\n"
                    f"{first_response}\n\n"
                    f'<tool_output tool="{action_name}">\n'
                    f"{tool_output}\n"
                    f"</tool_output>\n\n"
                    "Now provide the final response to the user based on the tool output above:"
                )
                final_response = self.llm.generate(followup_prompt)

        return {
            "response": final_response,
            "decision": decision,
            "tool_executed": tool_executed,
            "tool_output": tool_output,
        }
