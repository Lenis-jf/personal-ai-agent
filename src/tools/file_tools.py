"""Herramientas locales para lectura de documentos multiformato y metadatos de medios."""
from __future__ import annotations

from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader
from docx import Document

# Colección inmutable de directorios permitidos
ALLOWED_BASE_DIRECTORIES = {
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
    Path.home() / "Proyectos",
    Path(__file__).resolve().parents[2] / "test_sandbox",
}

BLOCKED_PATTERNS = {
    ".env",
    ".git",
    ".ssh",
    ".gnupg",
    ".zsh_history",
    ".bash_history",
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "credentials",
    "secrets",
    ".netrc",
}

BLOCKED_EXTENSIONS = {
    ".pem",
    ".key",
    ".pfx",
    ".p12",
    ".crt",
    ".der",
}

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}


def _read_pdf(path: Path, max_chars: int) -> str:
    reader = PdfReader(str(path))
    pages_text = []
    total_len = 0

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages_text.append(f"--- Página {i + 1} ---\n{text}")
        total_len += len(text)
        if total_len >= max_chars:
            break

    full_text = "\n\n".join(pages_text)
    return full_text[:max_chars]


def _read_docx(path: Path, max_chars: int) -> str:
    doc = Document(str(path))
    paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
    full_text = "\n".join(paragraphs)
    return full_text[:max_chars]


def _inspect_image(path: Path) -> str:
    with Image.open(path) as img:
        width, height = img.size
        img_format = img.format or path.suffix.upper().replace(".", "")
        size_kb = path.stat().st_size / 1024
        return (
            f"[Archivo de Imagen Detectado]\n"
            f"- Tipo: {img_format}\n"
            f"- Dimensiones: {width}x{height} píxeles\n"
            f"- Tamaño: {size_kb:.1f} KB\n"
            f"- Ubicación: {path}\n"
            f"Nota para el asistente: El sistema no cuenta con visión artificial activa. "
            f"Informa al usuario sobre estos metadatos y proporciónale la ruta absoluta si desea abrirla."
        )


def _inspect_svg(path: Path) -> str:
    tree = ET.parse(path)
    root = tree.getroot()
    width = root.attrib.get("width", "desconocido")
    height = root.attrib.get("height", "desconocido")
    viewbox = root.attrib.get("viewBox", "no especificado")
    size_kb = path.stat().st_size / 1024
    return (
        f"[Archivo Vectorial SVG Detectado]\n"
        f"- Dimensiones: width={width}, height={height}, viewBox={viewbox}\n"
        f"- Tamaño: {size_kb:.1f} KB\n"
        f"- Ubicación: {path}\n"
        f"Nota para el asistente: Informa al usuario sobre los metadatos vectoriales disponibles."
    )


def safe_read_file(filepath: str, max_chars: int = 12_000) -> str:
    """Lee documentos de texto, PDF, DOCX o inspecciona metadatos de imágenes/SVG."""
    try:
        raw_path = Path(filepath).expanduser()

        if raw_path.name.startswith("."):
            return f"Error de seguridad: Acceso denegado a archivo oculto ('{raw_path.name}')."

        target = raw_path.resolve()

        if not any(target.is_relative_to(base.resolve()) for base in ALLOWED_BASE_DIRECTORIES):
            return (
                f"Error de seguridad: Acceso denegado. "
                f"La ruta '{filepath}' está fuera de las carpetas autorizadas."
            )

        target_name_lower = target.name.lower()
        if any(blocked in target_name_lower for blocked in BLOCKED_PATTERNS):
            return f"Error de seguridad: Acceso denegado a archivo protegido ('{target.name}')."

        suffix = target.suffix.lower()
        if suffix in BLOCKED_EXTENSIONS:
            return f"Error de seguridad: Acceso denegado a certificados o claves ('{suffix}')."

        if not target.exists():
            return f"Error: El archivo '{filepath}' no existe."

        if not target.is_file():
            return f"Error: La ruta '{filepath}' no corresponde a un archivo regular."

        # Ruteo según el tipo de archivo
        if suffix == ".pdf":
            content = _read_pdf(target, max_chars)
        elif suffix in {".docx", ".doc"}:
            content = _read_docx(target, max_chars)
        elif suffix in IMAGE_EXTENSIONS:
            return _inspect_image(target)
        elif suffix == ".svg":
            return _inspect_svg(target)
        else:
            # Archivo de texto plano general
            content = target.read_text(encoding="utf-8", errors="replace")
            if len(content) > max_chars:
                content = content[:max_chars] + f"\n... [Contenido truncado a {max_chars} caracteres]"

        return content

    except Exception as exc:
        return f"Error al procesar el archivo '{filepath}': {exc}"


TOOL_METADATA: dict[str, dict[str, Any]] = {
    "file_read": {
        "function": safe_read_file,
        "description": "Reads text content from plain text (.txt, .md, .py, .json), PDF (.pdf), Word (.docx), or inspects image/SVG files.",
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {
                    "type": "string",
                    "description": "Path to the target file (e.g. '~/Downloads/document.pdf' or 'test_sandbox/notes.txt')",
                }
            },
            "required": ["filepath"],
        },
    }
}
