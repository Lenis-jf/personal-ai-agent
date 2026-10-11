"""Generate the v1 multilingual router benchmark (human-labelled, evaluation only)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.router.schema import TOOL_FIELDS, question_schema

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / "routing_v1.jsonl"

# Intentional policy: cloud_candidate is only a recommendation. It never authorizes
# transmission of local/private content or enables a cloud provider by itself.
# Each language contains 18 naturally worded examples: three per skill. Labels are
# policy-based gold labels, not objective truths. Keep the benchmark frozen once run.
CASES = [
    # SPANISH
    {"id":"es-coding-01","language":"es","category":"coding","request":"Explícame qué hace una comprensión de listas en Python con un ejemplo corto.","skill":"coding","complexity":"simple","model_route":"local"},
    {"id":"es-coding-02","language":"es","category":"coding","request":"En mi proyecto, los tests fallan con un AttributeError. Revisa el código relacionado y ayúdame a localizar la causa.","skill":"coding","complexity":"moderate","model_route":"local","tool_file_search":True,"tool_file_read":True},
    {"id":"es-coding-03","language":"es","category":"coding","request":"Analiza la arquitectura completa de una aplicación con muchos módulos y diseña un plan de migración gradual con riesgos, alternativas y pruebas para cada fase.","skill":"coding","complexity":"complex","model_route":"cloud_candidate","tool_file_search":True,"tool_file_read":True},
    {"id":"es-study-01","language":"es","category":"study_assistant","request":"Explícame la recursión como si estuviera empezando a programar.","skill":"study_assistant","complexity":"simple","model_route":"local"},
    {"id":"es-study-02","language":"es","category":"study_assistant","request":"Tengo seis temas para un examen dentro de una semana. Hazme un plan de estudio realista con repasos y ejercicios.","skill":"study_assistant","complexity":"moderate","model_route":"local"},
    {"id":"es-study-03","language":"es","category":"study_assistant","request":"Compara en profundidad modelos de consistencia distribuida, memoria y concurrencia, y construye una guía de estudio avanzada con dependencias entre los conceptos.","skill":"study_assistant","complexity":"complex","model_route":"cloud_candidate"},
    {"id":"es-files-01","language":"es","category":"file_management","request":"Busca todos los archivos PDF que tengo en la carpeta Descargas.","skill":"file_management","complexity":"simple","model_route":"local","tool_file_search":True},
    {"id":"es-files-02","language":"es","category":"file_management","request":"Lee el PDF de la matrícula que está en mi carpeta Documentos y resúmeme sus requisitos.","skill":"file_management","complexity":"moderate","model_route":"local","tool_file_read":True},
    {"id":"es-files-03","language":"es","category":"file_management","request":"Encuentra archivos duplicados en Descargas, comprueba cuáles son idénticos y elimina los duplicados exactos.","skill":"file_management","complexity":"complex","model_route":"local","tool_file_search":True,"tool_file_read":True,"tool_file_modify":True,"requires_confirmation":True},
    {"id":"es-email-01","language":"es","category":"email_assistant","request":"Redacta un correo amable a mi profesor para pedir una aclaración sobre la fecha de entrega.","skill":"email_assistant","complexity":"simple","model_route":"local","tool_email_draft":True},
    {"id":"es-email-02","language":"es","category":"email_assistant","request":"Revisa este borrador en alemán para que suene natural, cordial y profesional, sin cambiar lo que quiero decir.","skill":"email_assistant","complexity":"moderate","model_route":"local","tool_email_draft":True},
    {"id":"es-email-03","language":"es","category":"email_assistant","request":"Envía este mensaje a la persona de recursos humanos: 'Confirmo que recibí los documentos. Gracias'. No he indicado su dirección de correo.","skill":"email_assistant","complexity":"moderate","model_route":"local","tool_email_send":True,"needs_clarification":True,"requires_confirmation":True},
    {"id":"es-research-01","language":"es","category":"research","request":"Busca en fuentes oficiales cuál es la versión estable más reciente de Python y cita las fuentes.","skill":"research","complexity":"moderate","model_route":"local","tool_web_search":True},
    {"id":"es-research-02","language":"es","category":"research","request":"Investiga y compara diez herramientas locales de agentes de IA usando documentación actual, licencias y pruebas técnicas; sintetiza las diferencias y las incertidumbres.","skill":"research","complexity":"complex","model_route":"cloud_candidate","tool_web_search":True},
    {"id":"es-research-03","language":"es","category":"research","request":"Encuentra la documentación oficial de una biblioteca que acaba de cambiar su API y resume qué cambió en la versión actual.","skill":"research","complexity":"moderate","model_route":"local","tool_web_search":True},
    {"id":"es-general-01","language":"es","category":"personal_assistant","request":"Dame tres ideas sencillas para preparar una cena con arroz y verduras.","skill":"personal_assistant","complexity":"simple","model_route":"local"},
    {"id":"es-general-02","language":"es","category":"personal_assistant","request":"Planifica cinco días de viaje equilibrando transporte, horarios de apertura, presupuesto y varias alternativas por si llueve.","skill":"personal_assistant","complexity":"complex","model_route":"cloud_candidate","tool_web_search":True},
    {"id":"es-general-03","language":"es","category":"personal_assistant","request":"Limpia la caché de mis aplicaciones y actualiza todo el software instalado en mi ordenador.","skill":"personal_assistant","complexity":"complex","model_route":"local","tool_system_change":True,"requires_confirmation":True,"needs_clarification":True},

    # ENGLISH
    {"id":"en-coding-01","language":"en","category":"coding","request":"Explain what a Python list comprehension does and show one short example.","skill":"coding","complexity":"simple","model_route":"local"},
    {"id":"en-coding-02","language":"en","category":"coding","request":"My project tests fail with an AttributeError. Inspect the relevant code and help me find the cause.","skill":"coding","complexity":"moderate","model_route":"local","tool_file_search":True,"tool_file_read":True},
    {"id":"en-coding-03","language":"en","category":"coding","request":"Review the architecture of a large application with many modules and design a phased migration plan including risks, alternatives, and tests for each phase.","skill":"coding","complexity":"complex","model_route":"cloud_candidate","tool_file_search":True,"tool_file_read":True},
    {"id":"en-study-01","language":"en","category":"study_assistant","request":"Explain recursion as if I were just starting to learn programming.","skill":"study_assistant","complexity":"simple","model_route":"local"},
    {"id":"en-study-02","language":"en","category":"study_assistant","request":"I have six topics to study for an exam in one week. Make a realistic study plan with review sessions and exercises.","skill":"study_assistant","complexity":"moderate","model_route":"local"},
    {"id":"en-study-03","language":"en","category":"study_assistant","request":"Compare distributed consistency models, memory, and concurrency in depth, then build an advanced study guide showing dependencies between concepts.","skill":"study_assistant","complexity":"complex","model_route":"cloud_candidate"},
    {"id":"en-files-01","language":"en","category":"file_management","request":"Find every PDF file in my Downloads folder.","skill":"file_management","complexity":"simple","model_route":"local","tool_file_search":True},
    {"id":"en-files-02","language":"en","category":"file_management","request":"Read the enrollment PDF in my Documents folder and summarize its requirements.","skill":"file_management","complexity":"moderate","model_route":"local","tool_file_read":True},
    {"id":"en-files-03","language":"en","category":"file_management","request":"Find duplicate files in Downloads, verify which ones are identical, and delete exact duplicates.","skill":"file_management","complexity":"complex","model_route":"local","tool_file_search":True,"tool_file_read":True,"tool_file_modify":True,"requires_confirmation":True},
    {"id":"en-email-01","language":"en","category":"email_assistant","request":"Draft a friendly email to my professor asking for clarification about the submission deadline.","skill":"email_assistant","complexity":"simple","model_route":"local","tool_email_draft":True},
    {"id":"en-email-02","language":"en","category":"email_assistant","request":"Edit this German draft so it sounds natural, polite, and professional without changing my meaning.","skill":"email_assistant","complexity":"moderate","model_route":"local","tool_email_draft":True},
    {"id":"en-email-03","language":"en","category":"email_assistant","request":"Send this message to my HR contact: 'I confirm that I received the documents. Thank you.' I have not provided their email address.","skill":"email_assistant","complexity":"moderate","model_route":"local","tool_email_send":True,"needs_clarification":True,"requires_confirmation":True},
    {"id":"en-research-01","language":"en","category":"research","request":"Find the latest stable Python version from official sources and cite the sources.","skill":"research","complexity":"moderate","model_route":"local","tool_web_search":True},
    {"id":"en-research-02","language":"en","category":"research","request":"Research and compare ten local AI-agent tools using current documentation, licenses, and technical tests; synthesize the differences and uncertainties.","skill":"research","complexity":"complex","model_route":"cloud_candidate","tool_web_search":True},
    {"id":"en-research-03","language":"en","category":"research","request":"Find the official documentation for a library whose API recently changed and summarize what changed in the current version.","skill":"research","complexity":"moderate","model_route":"local","tool_web_search":True},
    {"id":"en-general-01","language":"en","category":"personal_assistant","request":"Give me three simple dinner ideas using rice and vegetables.","skill":"personal_assistant","complexity":"simple","model_route":"local"},
    {"id":"en-general-02","language":"en","category":"personal_assistant","request":"Plan a five-day trip balancing transport, opening hours, budget, and backup options in case it rains.","skill":"personal_assistant","complexity":"complex","model_route":"cloud_candidate","tool_web_search":True},
    {"id":"en-general-03","language":"en","category":"personal_assistant","request":"Clear my application caches and update all software installed on my computer.","skill":"personal_assistant","complexity":"complex","model_route":"local","tool_system_change":True,"requires_confirmation":True,"needs_clarification":True},

    # GERMAN
    {"id":"de-coding-01","language":"de","category":"coding","request":"Erkläre mir, was eine List Comprehension in Python macht, mit einem kurzen Beispiel.","skill":"coding","complexity":"simple","model_route":"local"},
    {"id":"de-coding-02","language":"de","category":"coding","request":"In meinem Projekt schlagen Tests mit einem AttributeError fehl. Untersuche den relevanten Code und hilf mir, die Ursache zu finden.","skill":"coding","complexity":"moderate","model_route":"local","tool_file_search":True,"tool_file_read":True},
    {"id":"de-coding-03","language":"de","category":"coding","request":"Analysiere die Architektur einer großen Anwendung mit vielen Modulen und entwirf einen schrittweisen Migrationsplan mit Risiken, Alternativen und Tests für jede Phase.","skill":"coding","complexity":"complex","model_route":"cloud_candidate","tool_file_search":True,"tool_file_read":True},
    {"id":"de-study-01","language":"de","category":"study_assistant","request":"Erkläre Rekursion so, als würde ich gerade erst Programmieren lernen.","skill":"study_assistant","complexity":"simple","model_route":"local"},
    {"id":"de-study-02","language":"de","category":"study_assistant","request":"Ich muss sechs Themen für eine Prüfung in einer Woche lernen. Erstelle einen realistischen Lernplan mit Wiederholungen und Übungen.","skill":"study_assistant","complexity":"moderate","model_route":"local"},
    {"id":"de-study-03","language":"de","category":"study_assistant","request":"Vergleiche verteilte Konsistenzmodelle, Speicher und Nebenläufigkeit ausführlich und erstelle einen anspruchsvollen Lernleitfaden mit Abhängigkeiten zwischen den Konzepten.","skill":"study_assistant","complexity":"complex","model_route":"cloud_candidate"},
    {"id":"de-files-01","language":"de","category":"file_management","request":"Finde alle PDF-Dateien in meinem Downloads-Ordner.","skill":"file_management","complexity":"simple","model_route":"local","tool_file_search":True},
    {"id":"de-files-02","language":"de","category":"file_management","request":"Lies die Immatrikulationsbescheinigung als PDF in meinem Dokumente-Ordner und fasse die Anforderungen zusammen.","skill":"file_management","complexity":"moderate","model_route":"local","tool_file_read":True},
    {"id":"de-files-03","language":"de","category":"file_management","request":"Finde doppelte Dateien in Downloads, prüfe, welche wirklich identisch sind, und lösche exakte Duplikate.","skill":"file_management","complexity":"complex","model_route":"local","tool_file_search":True,"tool_file_read":True,"tool_file_modify":True,"requires_confirmation":True},
    {"id":"de-email-01","language":"de","category":"email_assistant","request":"Schreibe eine freundliche E-Mail an meinen Professor, um nach einer Klärung zur Abgabefrist zu fragen.","skill":"email_assistant","complexity":"simple","model_route":"local","tool_email_draft":True},
    {"id":"de-email-02","language":"de","category":"email_assistant","request":"Überarbeite diesen englischen Entwurf, damit er natürlich, höflich und professionell klingt, ohne die Bedeutung zu verändern.","skill":"email_assistant","complexity":"moderate","model_route":"local","tool_email_draft":True},
    {"id":"de-email-03","language":"de","category":"email_assistant","request":"Sende diese Nachricht an meine HR-Ansprechperson: 'Ich bestätige den Erhalt der Dokumente. Vielen Dank.' Die E-Mail-Adresse habe ich noch nicht angegeben.","skill":"email_assistant","complexity":"moderate","model_route":"local","tool_email_send":True,"needs_clarification":True,"requires_confirmation":True},
    {"id":"de-research-01","language":"de","category":"research","request":"Finde die neueste stabile Python-Version anhand offizieller Quellen und nenne die Quellen.","skill":"research","complexity":"moderate","model_route":"local","tool_web_search":True},
    {"id":"de-research-02","language":"de","category":"research","request":"Recherchiere und vergleiche zehn lokale KI-Agenten-Tools anhand aktueller Dokumentation, Lizenzen und technischer Tests. Fasse Unterschiede und Unsicherheiten zusammen.","skill":"research","complexity":"complex","model_route":"cloud_candidate","tool_web_search":True},
    {"id":"de-research-03","language":"de","category":"research","request":"Finde die offizielle Dokumentation einer Bibliothek, deren API kürzlich geändert wurde, und fasse die Änderungen der aktuellen Version zusammen.","skill":"research","complexity":"moderate","model_route":"local","tool_web_search":True},
    {"id":"de-general-01","language":"de","category":"personal_assistant","request":"Gib mir drei einfache Abendessen-Ideen mit Reis und Gemüse.","skill":"personal_assistant","complexity":"simple","model_route":"local"},
    {"id":"de-general-02","language":"de","category":"personal_assistant","request":"Plane fünf Reisetage unter Berücksichtigung von Verkehr, Öffnungszeiten, Budget und Alternativen bei Regen.","skill":"personal_assistant","complexity":"complex","model_route":"cloud_candidate","tool_web_search":True},
    {"id":"de-general-03","language":"de","category":"personal_assistant","request":"Leere die Caches meiner Anwendungen und aktualisiere sämtliche installierte Software auf meinem Computer.","skill":"personal_assistant","complexity":"complex","model_route":"local","tool_system_change":True,"requires_confirmation":True,"needs_clarification":True},
]


def build_rows() -> list[dict]:
    rows = []
    for case in CASES:
        lang = case["language"]
        labels = {
            "skill": case["skill"],
            "complexity": case["complexity"],
            "model_route": case["model_route"],
        }
        for field in TOOL_FIELDS + ("needs_clarification", "requires_confirmation"):
            labels[field] = "yes" if case.get(field, False) else "no"
        targets = {field: {"label": label} for field, label in labels.items()}
        rows.append({
            "id": case["id"],
            "language": lang,
            "category": case["category"],
            "state": {"user_request": case["request"]},
            "questions": question_schema(lang),
            "targets": targets,
        })
    return rows


def main() -> None:
    rows = build_rows()
    OUTPUT.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    counts = {}
    for row in rows:
        counts[row["language"]] = counts.get(row["language"], 0) + 1
    print(f"Wrote {len(rows)} cases to {OUTPUT}")
    print("By language:", counts)
    print(f"Decisions per case: {len(rows[0]['questions'])}; total target decisions: {len(rows) * len(rows[0]['questions'])}")


if __name__ == "__main__":
    main()
