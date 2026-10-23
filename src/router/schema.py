"""Decision schema for runtime routing and two-stage modular sub-skill selection."""
from __future__ import annotations

SKILL_OPTIONS = {
    "coding": "Writing, understanding, debugging, inspecting source files, or locating software code and functions",
    "file_management": "Finding, reading, summarizing, visually inspecting images/SVGs, or exploring local files and directories",
    "personal_assistant": "General assistance, translations, or conversation using existing session memory not covered by other skills",
}

SUB_SKILL_OPTIONS = {
    "file_locator": "Locating files or folders by name, or exploring directory tree structures (file_search, dir_tree)",
    "document_reader": "Reading, extracting text, or summarizing documents like PDF, DOCX, TXT, MD, or JSON/JS data files (file_read)",
    "code_inspector": "Reading source code files and formatting code blocks with exact file paths and line numbers",
    "content_grep": "Searching inside files for specific function definitions, classes, variables, or exact text phrases (file_grep)",
    "image_vision": "Inspecting images (.png, .jpg, .webp) with Computer Vision, reading image metadata, or inspecting .svg vector files",
}

COMPLEXITY_OPTIONS = {
    "simple": "One straightforward step or a common explanation",
    "moderate": "Several steps or some contextual analysis",
    "advanced": "Complex reasoning, multiple sources, or intricate problem-solving",
    "complex": "Many dependent steps, deep analysis, or synthesis across many sources",
}

MODEL_ROUTE_OPTIONS = {
    "local": "Use the local model by default, especially for routine or private tasks",
    "cloud_candidate": "A stronger cloud model may help after privacy review and explicit user approval",
}

BOOLEAN_QUESTIONS = {
    # Context dependency check (triggers Stage-2 with previous turn history if True)
    "needs_previous_context": (
        "Does the user's message refer implicitly to a previous turn (e.g., 'that file', "
        "'the same folder', 'translate it', 'search there again') without naming the target explicitly?"
    ),
    # Tool activation predictions
    "tool_file_search": "Does the task require searching for files or folders by name or keyword?",
    "tool_dir_tree": "Does the task require listing or exploring the folder structure of a directory?",
    "tool_file_read": "Does the task require reading the contents of a file, document, code file, or image?",
    "tool_file_grep": "Does the task require searching inside files for specific text, code, or function definitions?",
    # Modular Sub-Skill activation predictions
    "sub_file_locator": "Does the task involve locating files or exploring directory trees on the computer?",
    "sub_document_reader": "Does the task involve reading or summarizing text documents, PDFs, DOCX, or data files?",
    "sub_code_inspector": "Does the task involve reading, displaying, or explaining source code files (.py, .js, .jsx, .java, .c)?",
    "sub_content_grep": "Does the task involve searching inside files for a specific function definition, variable, or text phrase?",
    "sub_image_vision": "Does the task involve images (.png, .jpg, .webp), vector graphics (.svg), or visual inspection with Computer Vision?",
    # Safety & clarification flags
    "needs_clarification": "Is essential information missing such that the agent should ask a clarifying question before proceeding?",
    "requires_confirmation": "Must the agent obtain explicit confirmation before an external, destructive, or consequential action?",
}

TOOL_FIELDS = (
    "tool_file_search",
    "tool_dir_tree",
    "tool_file_read",
    "tool_file_grep",
)

SUB_SKILL_FIELDS = (
    "sub_file_locator",
    "sub_document_reader",
    "sub_code_inspector",
    "sub_content_grep",
    "sub_image_vision",
)


def question_schema(language: str = "en") -> dict:
    """Build the primary classification schema (Stage 1) for Intern-Decision."""
    questions = {
        "skill": {
            "type": "choice",
            "instructions": "Which primary skill best matches the user's main intent?",
            "criteria": SKILL_OPTIONS,
        },
        "complexity": {
            "type": "choice",
            "instructions": "How demanding is the reasoning and execution needed?",
            "criteria": COMPLEXITY_OPTIONS,
        },
        "model_route": {
            "type": "choice",
            "instructions": (
                "Which model route should be considered for this task? "
                "A cloud candidate is only a recommendation; private data must not be sent without explicit approval."
            ),
            "criteria": MODEL_ROUTE_OPTIONS,
        },
    }
    for field, instruction in BOOLEAN_QUESTIONS.items():
        questions[field] = {"type": "noul", "instructions": instruction}
    return questions


def focused_subskill_schema(candidate_skill: str = "file_management") -> dict:
    """
    Build the Stage-2 focused re-routing schema when Stage 1 detected implicit follow-up context,
    had low confidence (<0.55), or failed to select any sub-skill module.
    """
    questions: dict = {
        "skill": {
            "type": "choice",
            "instructions": (
                "Given the conversation context and current user request, confirm which primary skill "
                "is most appropriate for this turn:"
            ),
            "criteria": SKILL_OPTIONS,
        },
        "primary_sub_skill": {
            "type": "choice",
            "instructions": (
                f"For the '{candidate_skill}' workflow, which specialized sub-skill module "
                "is the single most important capability needed right now?"
            ),
            "criteria": SUB_SKILL_OPTIONS,
        },
    }
    for field, instruction in BOOLEAN_QUESTIONS.items():
        if field != "needs_previous_context":
            questions[field] = {"type": "noul", "instructions": instruction}
    return questions