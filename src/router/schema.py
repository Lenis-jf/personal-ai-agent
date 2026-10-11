"""Shared decision schema for runtime routing and the multilingual benchmark."""
from __future__ import annotations

SKILL_TEXT = {
    "en": {
        "question": "Which skill best matches the user's main intent?",
        "options": {
            "coding": "Writing, understanding, debugging, or reviewing software code",
            "study_assistant": "Learning, explaining concepts, studying, or preparing for an exam",
            "file_management": "Finding, reading, organizing, modifying, or deleting local files",
            "email_assistant": "Drafting, revising, or sending emails and messages",
            "research": "Finding current external information or comparing sources",
            "personal_assistant": "General assistance or planning not covered by the other skills",
        },
    },
    "es": {
        "question": "¿Qué skill coincide mejor con la intención principal del usuario?",
        "options": {
            "coding": "Escribir, entender, depurar o revisar código de software",
            "study_assistant": "Aprender, explicar conceptos, estudiar o preparar un examen",
            "file_management": "Buscar, leer, organizar, modificar o eliminar archivos locales",
            "email_assistant": "Redactar, revisar o enviar correos y mensajes",
            "research": "Buscar información externa actual o comparar fuentes",
            "personal_assistant": "Ayuda general o planificación que no encaja en las otras skills",
        },
    },
    "de": {
        "question": "Welche Skill passt am besten zur Hauptabsicht der Anfrage?",
        "options": {
            "coding": "Softwarecode schreiben, verstehen, debuggen oder überprüfen",
            "study_assistant": "Lernen, Konzepte erklären, studieren oder sich auf eine Prüfung vorbereiten",
            "file_management": "Lokale Dateien suchen, lesen, organisieren, ändern oder löschen",
            "email_assistant": "E-Mails und Nachrichten verfassen, überarbeiten oder senden",
            "research": "Aktuelle externe Informationen suchen oder Quellen vergleichen",
            "personal_assistant": "Allgemeine Hilfe oder Planung, die zu keiner anderen Skill passt",
        },
    },
}

QUESTION_TEXT = {
    "en": {
        "complexity": "How demanding is the reasoning and execution needed?",
        "complexity_options": {
            "simple": "One straightforward step or a common explanation",
            "moderate": "Several steps or some contextual analysis",
            "complex": "Many dependent steps, deep analysis, or synthesis across many sources",
        },
        "model_route": "Which model route should be considered for this task? A cloud candidate is only a recommendation; private data must not be sent without explicit approval.",
        "model_route_options": {
            "local": "Use the local model by default, especially for routine or private tasks",
            "cloud_candidate": "A stronger cloud model may help after privacy review and explicit user approval",
        },
        "tool_file_search": "Does the task require searching or listing files on the user's computer?",
        "tool_file_read": "Does the task require reading the contents of local files?",
        "tool_file_modify": "Does the task require creating, editing, moving, renaming, or deleting local files?",
        "tool_web_search": "Does the task require searching the web or checking current external information?",
        "tool_email_draft": "Does the task require drafting or revising an email or message?",
        "tool_email_send": "Does the task explicitly require sending an email or message to someone?",
        "tool_system_change": "Does the task require changing system settings, installing/updating software, or cleaning system/app data?",
        "needs_clarification": "Is essential information missing such that the agent should ask a clarifying question before proceeding?",
        "requires_confirmation": "Must the agent obtain explicit confirmation before an external, destructive, or consequential action?",
    },
    "es": {
        "complexity": "¿Qué nivel de razonamiento y ejecución requiere la tarea?",
        "complexity_options": {
            "simple": "Un paso directo o una explicación habitual",
            "moderate": "Varios pasos o cierto análisis del contexto",
            "complex": "Muchos pasos dependientes, análisis profundo o síntesis de numerosas fuentes",
        },
        "model_route": "¿Qué ruta de modelo conviene considerar? Una opción cloud es solo una recomendación; no se deben enviar datos privados sin aprobación explícita.",
        "model_route_options": {
            "local": "Usar el modelo local por defecto, especialmente en tareas rutinarias o privadas",
            "cloud_candidate": "Un modelo cloud más potente podría ayudar tras revisar la privacidad y obtener aprobación explícita",
        },
        "tool_file_search": "¿La tarea requiere buscar o listar archivos en el ordenador del usuario?",
        "tool_file_read": "¿La tarea requiere leer el contenido de archivos locales?",
        "tool_file_modify": "¿La tarea requiere crear, editar, mover, renombrar o eliminar archivos locales?",
        "tool_web_search": "¿La tarea requiere buscar en la web o consultar información externa actualizada?",
        "tool_email_draft": "¿La tarea requiere redactar o revisar un correo o mensaje?",
        "tool_email_send": "¿La tarea pide explícitamente enviar un correo o mensaje a otra persona?",
        "tool_system_change": "¿La tarea requiere cambiar ajustes del sistema, instalar/actualizar software o limpiar datos del sistema o de aplicaciones?",
        "needs_clarification": "¿Falta información esencial y el agente debería hacer una pregunta aclaratoria antes de continuar?",
        "requires_confirmation": "¿El agente debe obtener confirmación explícita antes de una acción externa, destructiva o importante?",
    },
    "de": {
        "complexity": "Wie anspruchsvoll sind die erforderlichen Überlegungen und Handlungsschritte?",
        "complexity_options": {
            "simple": "Ein klarer Schritt oder eine gewöhnliche Erklärung",
            "moderate": "Mehrere Schritte oder eine gewisse Kontextanalyse",
            "complex": "Viele voneinander abhängige Schritte, tiefgehende Analyse oder Synthese zahlreicher Quellen",
        },
        "model_route": "Welche Modellroute sollte erwogen werden? Cloud ist nur eine Empfehlung; private Daten dürfen nicht ohne ausdrückliche Zustimmung übertragen werden.",
        "model_route_options": {
            "local": "Standardmäßig das lokale Modell verwenden, besonders bei Routineaufgaben oder privaten Daten",
            "cloud_candidate": "Ein stärkeres Cloud-Modell könnte nach Datenschutzprüfung und ausdrücklicher Zustimmung helfen",
        },
        "tool_file_search": "Muss der Agent Dateien auf dem Computer des Nutzers suchen oder auflisten?",
        "tool_file_read": "Muss der Agent Inhalte lokaler Dateien lesen?",
        "tool_file_modify": "Muss der Agent lokale Dateien erstellen, bearbeiten, verschieben, umbenennen oder löschen?",
        "tool_web_search": "Muss der Agent im Web suchen oder aktuelle externe Informationen prüfen?",
        "tool_email_draft": "Muss der Agent eine E-Mail oder Nachricht verfassen oder überarbeiten?",
        "tool_email_send": "Soll ausdrücklich eine E-Mail oder Nachricht an eine andere Person gesendet werden?",
        "tool_system_change": "Muss der Agent Systemeinstellungen ändern, Software installieren/aktualisieren oder System-/App-Daten bereinigen?",
        "needs_clarification": "Fehlen wesentliche Angaben, sodass der Agent vor dem Fortfahren nachfragen sollte?",
        "requires_confirmation": "Muss der Agent vor einer externen, destruktiven oder folgenreichen Aktion eine ausdrückliche Bestätigung einholen?",
    },
}

TOOL_FIELDS = (
    "tool_file_search",
    "tool_file_read",
    "tool_file_modify",
    "tool_web_search",
    "tool_email_draft",
    "tool_email_send",
    "tool_system_change",
)

def question_schema(language: str) -> dict:
    skill = SKILL_TEXT[language]
    text = QUESTION_TEXT[language]
    questions = {
        "skill": {"type": "choice", "instructions": skill["question"], "criteria": skill["options"]},
        "complexity": {"type": "choice", "instructions": text["complexity"], "criteria": text["complexity_options"]},
        "model_route": {"type": "choice", "instructions": text["model_route"], "criteria": text["model_route_options"]},
    }
    for field in TOOL_FIELDS:
        questions[field] = {"type": "noul", "instructions": text[field]}
    questions["needs_clarification"] = {"type": "noul", "instructions": text["needs_clarification"]}
    questions["requires_confirmation"] = {"type": "noul", "instructions": text["requires_confirmation"]}
    return questions
