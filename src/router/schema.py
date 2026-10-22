"""Decision schema for runtime routing (English instructions over multilingual user input)."""
from __future__ import annotations

SKILL_OPTIONS = {
    "coding": "Writing, understanding, debugging, or locating software code and functions",
    "file_management": "Finding, reading, summarizing, exploring directories, modifying, or deleting local files and documents",
    # "email_assistant": "Drafting, revising, or sending emails and messages",
    # "research": "Finding current external information or comparing sources",
    "personal_assistant": "General assistance, translations, or conversation not covered by the other skills",
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
    "tool_file_search": "Does the task require searching or listing files or directories on the user's computer?",
    "tool_file_read": "Does the task require reading the contents of local files?",
    # "tool_file_modify": "Does the task require creating, editing, moving, renaming, or deleting local files?",
    # "tool_web_search": "Does the task require searching the web or checking current external information?",
    # "tool_email_draft": "Does the task require drafting or revising an email or message?",
    # "tool_email_send": "Does the task explicitly require sending an email or message to someone?",
    # "tool_system_change": "Does the task require changing system settings, installing/updating software, or cleaning system/app data?",
    "needs_clarification": "Is essential information missing such that the agent should ask a clarifying question before proceeding?",
    "requires_confirmation": "Must the agent obtain explicit confirmation before an external, destructive, or consequential action?",
}

TOOL_FIELDS = (
    "tool_file_search",
    "tool_file_read",
    "tool_file_grep",
    "tool_dir_tree",
    # "tool_file_modify",
    # "tool_web_search",
    # "tool_email_draft",
    # "tool_email_send",
    # "tool_system_change",
)


def question_schema(language: str = "en") -> dict:
    """Build the classification schema for Intern-Decision."""
    questions = {
        "skill": {
            "type": "choice",
            "instructions": "Which skill best matches the user's main intent?",
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