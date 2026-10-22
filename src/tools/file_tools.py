"""Herramientas locales bajo el Principio de Responsabilidad Única (SRP) sin redundancia en salidas:
- file_search: búsqueda específica de archivos (entry_type='file') o carpetas (entry_type='dir') con filtro de extensión.
- dir_tree: exploración compacta con rutas relativas, resumen de subcarpetas densas y filtros (extension_filter, sub_dir).
- file_grep: búsqueda de código/texto con extracción dinámica de funciones completas vía AST/llaves.
- file_read: lectura de archivos (soporta rutas absolutas, relativas al proyecto o rangos start_line..end_line).
"""
from __future__ import annotations

import ast
from collections import Counter
from difflib import SequenceMatcher
import fnmatch
import os
import re
from pathlib import Path
from typing import Any
import unicodedata
import xml.etree.ElementTree as ET

from PIL import Image
from pypdf import PdfReader
from docx import Document

ALLOWED_BASE_DIRECTORIES = {
    Path.home() / "Documents",
    Path.home() / "Desktop",
    Path.home() / "Downloads",
    Path.home() / "Documents" / "Proyectos",
    Path(__file__).resolve().parents[2] / "test_sandbox",
}

BLOCKED_PATTERNS = {
    ".env", ".git", ".ssh", ".gnupg", ".zsh_history", ".bash_history",
    "id_rsa", "id_ed25519", "id_ecdsa", "credentials", "secrets", ".netrc",
}

BLOCKED_EXTENSIONS = {
    ".pem", ".key", ".pfx", ".p12", ".crt", ".der",
}

IGNORE_DIR_NAMES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__",
    ".cache", ".build", "dist", "build", ".next", ".idea", ".vscode", "coverage", "logs",
}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff"}
MAX_GREP_FILE_BYTES = 1_000_000


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFC", text).casefold().strip()


def _sanitize_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _is_path_allowed(target: Path) -> bool:
    resolved = target.resolve()
    return any(resolved.is_relative_to(base.resolve()) for base in ALLOWED_BASE_DIRECTORIES)


def _get_search_roots() -> list[Path]:
    existing = [d for d in ALLOWED_BASE_DIRECTORIES if d.exists()]
    roots: list[Path] = []
    for d in existing:
        if not any(d != other and d.is_relative_to(other) for other in existing):
            roots.append(d)
    return roots


def _parse_extension_filter(ext_filter: str) -> set[str]:
    """Convierte una cadena como '.js, .json, jsx' en {'.js', '.json', '.jsx'}."""
    if not ext_filter or not ext_filter.strip():
        return set()
    parts = [p.strip().lower() for p in re.split(r"[,;\s]+", ext_filter) if p.strip()]
    return {p if p.startswith(".") else f".{p}" for p in parts}


def _fuzzy_token_match(query_tokens: list[str], candidate_name: str) -> float:
    norm_cand = _normalize(candidate_name)
    cand_parts = [p for p in re.split(r"[^a-z0-9áéíóúüñ]+", norm_cand) if p]
    if not cand_parts:
        return 0.0

    best_score = 0.0
    matched_tokens = 0

    for q_tok in query_tokens:
        if q_tok in norm_cand:
            matched_tokens += 1
            best_score = max(best_score, 1.0)
            continue

        tok_best = max(
            (SequenceMatcher(None, q_tok, c_part).ratio() for c_part in cand_parts),
            default=0.0,
        )
        if tok_best >= 0.72:
            matched_tokens += 1
            best_score = max(best_score, tok_best)

    if matched_tokens == 0:
        return 0.0
    return best_score + (0.2 * (matched_tokens - 1))


def _find_entries_by_name(
    pattern: str,
    search_roots: list[Path],
    max_results: int = 20,
    entry_type: str = "all",
    allowed_exts: set[str] | None = None,
) -> list[tuple[str, Path]]:
    raw_pattern = pattern.strip().strip("'\"")
    norm_pattern = _normalize(raw_pattern)
    has_wildcards = any(c in norm_pattern for c in ("*", "?", "["))
    glob_pattern = norm_pattern if has_wildcards else f"*{norm_pattern}*"

    mode = entry_type.lower().strip() if entry_type else "all"
    want_dirs = mode in {"all", "dir", "directory", "folder"}
    want_files = mode in {"all", "file", "files"}
    exts = allowed_exts or set()

    exact_matches: list[tuple[str, Path]] = []
    all_candidates: list[tuple[str, Path]] = []

    for root_dir in search_roots:
        for root, dirs, files in os.walk(root_dir, followlinks=False):
            dirs[:] = [d for d in dirs if not (d in IGNORE_DIR_NAMES or d.startswith("."))]

            if want_dirs and not exts:
                for d in dirs:
                    full_d = Path(root) / d
                    all_candidates.append(("DIR", full_d))
                    norm_d = _normalize(d)
                    if fnmatch.fnmatch(norm_d, glob_pattern) or (not has_wildcards and norm_pattern in norm_d):
                        exact_matches.append(("DIR", full_d))
                        if len(exact_matches) >= max_results:
                            return exact_matches

            if want_files:
                for file in files:
                    if file.startswith("."):
                        continue
                    full_f = Path(root) / file
                    if exts and full_f.suffix.lower() not in exts:
                        continue
                    all_candidates.append(("FILE", full_f))
                    norm_file = _normalize(file)
                    if fnmatch.fnmatch(norm_file, glob_pattern) or (not has_wildcards and norm_pattern in norm_file):
                        exact_matches.append(("FILE", full_f))
                        if len(exact_matches) >= max_results:
                            return exact_matches

    if exact_matches:
        return exact_matches

    generic_words = {"archivo", "archivos", "file", "files", "folder", "carpeta", "directorio", "web", "app", "code", "codigo"}
    query_tokens = [
        t for t in re.split(r"[^a-z0-9áéíóúüñ]+", norm_pattern)
        if len(t) >= 3 and t not in generic_words
    ]
    if not query_tokens:
        return []

    scored: list[tuple[float, str, Path]] = []
    for kind, path_obj in all_candidates:
        score = _fuzzy_token_match(query_tokens, path_obj.name)
        if score >= 0.72:
            bonus = 0.05 if kind == "DIR" else 0.0
            scored.append((score + bonus, kind, path_obj))

    scored.sort(key=lambda item: (-item[0], len(str(item[2]))))
    return [(kind, path_obj) for _, kind, path_obj in scored[:max_results]]


def _resolve_flexible_path(raw_input: str, expect_dir: bool = False) -> tuple[Path | None, list[Path], str]:
    """
    Resuelve rutas absolutas, rutas con ~, o rutas relativas a cualquiera de los directorios autorizados
    o subproyectos (ej. 'portfolio-2.0-vite/data/projectsData.js' o 'data/projectsData.js').
    """
    cleaned = raw_input.strip().strip("'\"")
    p = Path(cleaned).expanduser()
    if p.is_absolute() and p.exists():
        return p.resolve(), [], ""

    for base in [Path.home() / "Documents" / "Proyectos", *ALLOWED_BASE_DIRECTORIES]:
        if base.exists():
            cand = (base / cleaned).resolve()
            if cand.exists():
                return cand, [], ""

    target_name = p.name or cleaned
    entry_mode = "dir" if expect_dir else "all"
    found_entries = _find_entries_by_name(target_name, _get_search_roots(), max_results=8, entry_type=entry_mode)

    norm_cleaned = _normalize(cleaned.lstrip("./"))
    matching_suffix = [
        path_obj for _, path_obj in found_entries
        if _normalize(str(path_obj)).endswith(norm_cleaned)
    ]
    candidates = matching_suffix if matching_suffix else [path_obj for _, path_obj in found_entries]
    exact_name = [c for c in candidates if _normalize(c.name) == _normalize(target_name)]
    chosen = exact_name if exact_name else candidates

    if len(chosen) == 1:
        resolved = chosen[0].resolve()
        note = f"[Ruta resuelta automáticamente: {resolved}]\n"
        return resolved, [], note
    if len(chosen) > 1:
        return None, chosen, ""
    return p.resolve(), [], ""


def safe_dir_tree(
    dirpath: str,
    extension_filter: str = "",
    sub_dir: str = "",
    max_depth: int = 3,
    max_entries: int = 60,
) -> str:
    """
    Explora el árbol de un directorio usando rutas relativas limpias (sin repetir la ruta base en cada línea).
    No oculta ningún tipo de archivo: permite filtrar por `extension_filter` o `sub_dir`, y resume carpetas
    muy densas indicando qué tipos de archivos contienen.
    """
    try:
        target, multiples, auto_note = _resolve_flexible_path(dirpath, expect_dir=True)
        if multiples:
            opts = "\n".join(f"- {m}" for m in multiples)
            return f"Se encontraron varios directorios para '{dirpath}':\n{opts}\nEspecifica cuál deseas explorar."
        if target is None or not target.exists():
            return f"Error: El directorio '{dirpath}' no existe."

        if sub_dir and sub_dir.strip():
            sub_target = (target / sub_dir.strip().strip("/")).resolve()
            if not sub_target.exists():
                return f"Error: La subcarpeta '{sub_dir}' no existe dentro de '{target}'."
            target = sub_target

        if not _is_path_allowed(target):
            return f"Error de seguridad: Acceso denegado fuera de directorios autorizados ('{target}')."
        if not target.is_dir():
            return f"Error: '{target}' es un archivo, no un directorio. Usa `file_read` para leerlo."

        exts = _parse_extension_filter(extension_filter)
        depth_limit = max(1, min(int(max_depth or 3), 5))
        filter_info = f" | Filtro: {sorted(exts)}" if exts else ""
        lines = [
            f"{auto_note}[Directorio Base: {target}{filter_info}]",
            "(Rutas relativas al Directorio Base — puedes pasar `'nombre_carpeta/archivo'` o la ruta completa a `file_read`):",
        ]

        count = 0
        per_folder_preview = 60 if exts else 10

        for root, dirs, files in os.walk(target, followlinks=False):
            dirs[:] = sorted([d for d in dirs if not (d in IGNORE_DIR_NAMES or d.startswith("."))])
            rel_dir = Path(root).relative_to(target)
            depth = len(rel_dir.parts)
            if depth > depth_limit:
                dirs[:] = []
                continue

            visible_files = []
            for f in sorted(files):
                if f.startswith("."):
                    continue
                if exts and Path(f).suffix.lower() not in exts:
                    continue
                visible_files.append(f)

            indent = "  " * depth
            if depth > 0 and (visible_files or (not exts and dirs)):
                lines.append(f"{indent}📁 {rel_dir.as_posix()}/")

            shown_in_folder = visible_files[:per_folder_preview]
            hidden_in_folder = visible_files[per_folder_preview:]

            for f in shown_in_folder:
                rel_file = (rel_dir / f).as_posix() if depth > 0 else f
                lines.append(f"{indent}  📄 {rel_file}")
                count += 1
                if count >= max_entries:
                    lines.append(f"{indent}  ... [Límite de {max_entries} entradas alcanzado. Usa `sub_dir` o `extension_filter` para inspeccionar una sección específica]")
                    return "\n".join(lines)

            if hidden_in_folder:
                ext_counts = Counter(Path(f).suffix.lower() or "sin_ext" for f in hidden_in_folder)
                ext_summary = ", ".join(f"{k} ({v})" for k, v in ext_counts.most_common(4))
                lines.append(
                    f"{indent}  ... (+{len(hidden_in_folder)} archivos adicionales en '{rel_dir.as_posix()}': {ext_summary} "
                    f"— usa sub_dir=\"{rel_dir.as_posix()}\" para listarlos todos)"
                )

        if count == 0:
            if exts:
                lines.append(f"  (No se encontraron archivos con las extensiones {sorted(exts)} en este directorio)")
            else:
                lines.append("  (Directorio vacío)")
        return "\n".join(lines)

    except Exception as exc:
        return f"Error al explorar el directorio '{dirpath}': {exc}"


def _read_pdf(path: Path, max_chars: int) -> str:
    reader = PdfReader(str(path))
    pages_text = []
    total_len = 0
    for i, page in enumerate(reader.pages):
        text = _sanitize_text(page.extract_text() or "")
        pages_text.append(f"--- Página {i + 1} (de {len(reader.pages)}) ---\n{text}")
        total_len += len(text)
        if total_len >= max_chars:
            break
    full_text = "\n\n".join(pages_text)
    return full_text[:max_chars]


def _read_docx(path: Path, max_chars: int) -> str:
    doc = Document(str(path))
    paragraphs = [_sanitize_text(p.text) for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)[:max_chars]


def _inspect_image(path: Path) -> str:
    with Image.open(path) as img:
        width, height = img.size
        img_format = img.format or path.suffix.upper().replace(".", "")
        size_kb = path.stat().st_size / 1024
        return (
            f"[Archivo de Imagen Detectado]\n"
            f"- Tipo: {img_format} | Dimensiones: {width}x{height} px | Tamaño: {size_kb:.1f} KB\n"
            f"- Ruta: {path}\n"
            f"Nota: Visión artificial no activa. Describe estos metadatos al usuario."
        )


def _inspect_svg(path: Path) -> str:
    tree = ET.parse(path)
    root = tree.getroot()
    width = root.attrib.get("width", "desconocido")
    height = root.attrib.get("height", "desconocido")
    size_kb = path.stat().st_size / 1024
    return (
        f"[Archivo Vectorial SVG Detectado]\n"
        f"- Dimensiones: {width}x{height} | Tamaño: {size_kb:.1f} KB\n"
        f"- Ruta: {path}"
    )


def safe_read_file(
    filepath: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_chars: int = 20_000,
) -> str:
    """
    Lee el contenido de un archivo aceptando rutas absolutas o relativas.
    Calibrado en 20.000 caracteres (~5.000 tokens, <90 MB de KV Cache en M4 16GB).
    """
    try:
        raw_clean = filepath.strip().strip("'\"")
        if Path(raw_clean).name.startswith("."):
            return f"Error de seguridad: Acceso denegado a elemento oculto ('{raw_clean}')."

        target, multiples, auto_discovered_note = _resolve_flexible_path(raw_clean, expect_dir=False)
        if multiples:
            paths_str = "\n".join(f"- {p}" for p in multiples)
            return (
                f"Se encontraron varios archivos que coinciden con '{filepath}':\n{paths_str}\n"
                "Por favor invoca `file_read` indicando la ruta completa deseada."
            )

        if target is None or not target.exists():
            return f"Error: El archivo '{filepath}' no fue encontrado en los directorios autorizados."

        if not _is_path_allowed(target):
            return f"Error de seguridad: Acceso denegado fuera de directorios autorizados ('{filepath}')."

        target_name_lower = target.name.lower()
        if any(blocked in target_name_lower for blocked in BLOCKED_PATTERNS):
            return f"Error de seguridad: Acceso denegado a elemento protegido ('{target.name}')."

        if target.is_dir():
            return (
                f"[Nota: '{target.name}' es un directorio; mostrando su estructura con `dir_tree`]\n"
                + safe_dir_tree(str(target))
            )

        suffix = target.suffix.lower()
        if suffix in BLOCKED_EXTENSIONS:
            return f"Error de seguridad: Acceso denegado a claves o certificados ('{suffix}')."

        if suffix == ".pdf":
            content = _read_pdf(target, max_chars)
        elif suffix in {".docx", ".doc"}:
            content = _read_docx(target, max_chars)
        elif suffix in IMAGE_EXTENSIONS:
            return auto_discovered_note + _inspect_image(target)
        elif suffix == ".svg":
            return auto_discovered_note + _inspect_svg(target)
        else:
            raw_text = _sanitize_text(target.read_text(encoding="utf-8", errors="replace"))
            all_lines = raw_text.splitlines()
            total_lines = len(all_lines)
            if start_line is not None or end_line is not None:
                s_idx = max(1, int(start_line or 1))
                e_idx = min(total_lines, int(end_line or (s_idx + 200)))
                sliced = [
                    f"Línea {i:4d} | {all_lines[i - 1]}"
                    for i in range(s_idx, e_idx + 1)
                ]
                content = (
                    f"[Mostrando líneas {s_idx} a {e_idx} de {total_lines} totales]\n"
                    + "\n".join(sliced)
                )
            else:
                content = raw_text

            if len(content) > max_chars:
                clipped_part = content[:max_chars]
                lines_shown = clipped_part.count("\n") + 1
                content = (
                    clipped_part
                    + f"\n... [Archivo truncado en línea {lines_shown} de {total_lines} totales ({max_chars} de {len(content)} caracteres). "
                    f"Usa `start_line={lines_shown + 1}` y `end_line={total_lines}` en `file_read` para leer los elementos restantes]."
                )

        header = f"[Archivo leído: {target}]\n"
        return auto_discovered_note + header + content

    except Exception as exc:
        return f"Error al leer '{filepath}': {exc}"


def safe_search_files(
    pattern: str,
    entry_type: str = "all",
    extension_filter: str = "",
    search_dir: str = "",
    max_results: int = 20,
) -> str:
    """
    Localiza archivos ([FILE]) y/o directorios ([DIR]) por nombre (exacto o difuso),
    permitiendo filtrar por tipo de entrada ('dir', 'file', 'all') y extensión.
    """
    try:
        base_path = Path(search_dir).expanduser().resolve() if search_dir else None
        if base_path and not _is_path_allowed(base_path):
            return f"Error de seguridad: El directorio '{search_dir}' no está permitido."

        exts = _parse_extension_filter(extension_filter)
        targets = [base_path] if (base_path and base_path.exists()) else _get_search_roots()
        matches = _find_entries_by_name(
            pattern,
            targets,
            max_results=max_results,
            entry_type=entry_type,
            allowed_exts=exts,
        )

        if not matches and base_path:
            remaining = [d for d in _get_search_roots() if d != base_path]
            matches = _find_entries_by_name(
                pattern,
                remaining,
                max_results=max_results,
                entry_type=entry_type,
                allowed_exts=exts,
            )

        if matches:
            res = [f"Elementos encontrados ({len(matches)}):"]
            for kind, p in matches:
                res.append(f"- [{kind}] {p}")
            return "\n".join(res)

        return (
            f"No se encontraron elementos para pattern='{pattern}' (entry_type='{entry_type}', extension_filter='{extension_filter}'). "
            "Intenta con una palabra clave más general o cambia los filtros."
        )

    except Exception as exc:
        return f"Error al buscar archivos: {exc}"


def _extract_python_ast_bounds(source_text: str, target_line_1based: int) -> tuple[int, int] | None:
    try:
        tree = ast.parse(source_text)
        best_span: tuple[int, int] | None = None
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start_l = getattr(node, "lineno", None)
                end_l = getattr(node, "end_lineno", None)
                if start_l is not None and end_l is not None:
                    if start_l == target_line_1based:
                        return (start_l - 1, end_l)
                    if start_l <= target_line_1based <= end_l:
                        if best_span is None or (end_l - start_l) < (best_span[1] - best_span[0]):
                            best_span = (start_l - 1, end_l)
        return best_span
    except Exception:
        return None


def _extract_dynamic_code_bounds(
    lines: list[str],
    idx: int,
    raw_content: str,
    suffix: str,
    context_lines: int,
) -> tuple[int, int, bool]:
    line = lines[idx]
    stripped = line.lstrip()
    is_def = stripped.startswith(("def ", "async def ", "class ", "function ", "const ", "export function ", "export const "))

    if suffix == ".py" and is_def:
        ast_bounds = _extract_python_ast_bounds(raw_content, idx + 1)
        if ast_bounds is not None:
            s_idx, e_idx = ast_bounds
            return s_idx, min(len(lines), e_idx), True

        base_indent = len(line) - len(stripped)
        max_scan = min(len(lines), idx + 100)
        end_idx = max_scan
        for c_idx in range(idx + 1, max_scan):
            cand = lines[c_idx]
            cand_s = cand.lstrip()
            if not cand_s or cand_s.startswith("#"):
                continue
            if (len(cand) - len(cand_s)) <= base_indent:
                end_idx = c_idx
                break
        return idx, end_idx, True

    if suffix in {".js", ".jsx", ".ts", ".tsx", ".java", ".c", ".cpp"} and is_def:
        brace_balance = 0
        opened = False
        max_scan = min(len(lines), idx + 100)
        for c_idx in range(idx, max_scan):
            brace_balance += lines[c_idx].count("{") - lines[c_idx].count("}")
            if "{" in lines[c_idx]:
                opened = True
            if opened and brace_balance <= 0:
                return idx, c_idx + 1, True
        return idx, min(len(lines), idx + max(15, context_lines)), True

    before = max(1, context_lines // 3)
    after = max(4, context_lines)
    return max(0, idx - before), min(len(lines), idx + after + 1), is_def


def safe_grep_text(
    query: str,
    search_dir: str = "",
    file_extension: str = "",
    context_lines: int = 12,
    max_matches: int = 4,
) -> str:
    try:
        base_path = Path(search_dir).expanduser().resolve() if search_dir else None
        if base_path and not _is_path_allowed(base_path):
            return f"Error de seguridad: El directorio '{search_dir}' no está autorizado."

        targets = [base_path] if (base_path and base_path.exists()) else _get_search_roots()
        def_findings: list[str] = []
        other_findings: list[str] = []
        norm_query = _normalize(query.strip().strip("'\""))
        exts = _parse_extension_filter(file_extension)
        ctx_window = max(3, min(int(context_lines or 12), 80))

        if not norm_query:
            return "Error: La consulta de búsqueda (query) está vacía."

        for root_dir in targets:
            for root, dirs, files in os.walk(root_dir, followlinks=False):
                dirs[:] = [d for d in dirs if not (d in IGNORE_DIR_NAMES or d.startswith("."))]

                for file in files:
                    if file.startswith("."):
                        continue

                    file_path = Path(root) / file
                    suffix = file_path.suffix.lower()
                    if exts and suffix not in exts:
                        continue
                    if suffix in (IMAGE_EXTENSIONS | BLOCKED_EXTENSIONS | {".pdf", ".docx", ".pyc", ".lock", ".map"}):
                        continue

                    try:
                        if file_path.stat().st_size > MAX_GREP_FILE_BYTES:
                            continue
                        raw_content = _sanitize_text(file_path.read_text(encoding="utf-8", errors="ignore"))
                        lines = raw_content.splitlines()
                        for idx, line in enumerate(lines):
                            if norm_query in _normalize(line):
                                line_num = idx + 1
                                start_c, end_c, is_definition = _extract_dynamic_code_bounds(
                                    lines, idx, raw_content, suffix, ctx_window
                                )

                                tag = f"[DEFINICIÓN COMPLETA: líneas {start_c + 1} a {end_c}]" if is_definition else "[REFERENCIA]"
                                block_lines = [
                                    f"Archivo: {file_path} | Coincidencia exacta en LÍNEA {line_num} {tag}:"
                                ]
                                for c_idx in range(start_c, end_c):
                                    code_line = lines[c_idx][:200]
                                    block_lines.append(f"Línea {c_idx + 1:4d} | {code_line}")

                                block_str = "\n".join(block_lines)
                                if is_definition:
                                    def_findings.append(block_str)
                                else:
                                    other_findings.append(block_str)

                                if len(def_findings) >= max_matches:
                                    break
                    except Exception:
                        continue

        combined = def_findings[:max_matches] if def_findings else other_findings[:max_matches]
        if not combined:
            return f"No se encontraron coincidencias para '{query}' en los archivos."

        header = f"Coincidencias encontradas ({len(combined)} bloques):"
        return header + "\n\n" + "\n\n".join(combined)

    except Exception as exc:
        return f"Error al buscar texto en archivos: {exc}"


TOOL_METADATA: dict[str, dict[str, Any]] = {
    "file_search": {
        "function": safe_search_files,
        "description": (
            "Locates files ([FILE]) or directories ([DIR]) across authorized workspaces by name/keyword. "
            "Use entry_type='dir' when looking for a project folder, or entry_type='file' when looking for a document."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Single keyword or filename pattern (e.g., 'portfolio', 'StudentAngemeldetePruefung*.pdf')."},
                "entry_type": {"type": "string", "enum": ["all", "dir", "file"], "description": "Filter by 'dir' (folders only), 'file' (files only), or 'all'."},
                "extension_filter": {"type": "string", "description": "Optional extension filter (e.g., '.pdf', '.js,.jsx')."},
            },
            "required": ["pattern"],
        },
    },
    "dir_tree": {
        "function": safe_dir_tree,
        "description": (
            "Lists the internal structure of a directory using clean relative paths. "
            "Supports optional extension_filter (e.g. '.js,.json' or '.png,.pdf') and sub_dir to narrow down results."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "dirpath": {"type": "string", "description": "Path or name of the directory to explore."},
                "extension_filter": {"type": "string", "description": "Optional comma-separated extensions to filter (e.g. '.js,.jsx,.json' or '.png,.svg'). Omit to see all file types."},
                "sub_dir": {"type": "string", "description": "Optional subfolder inside dirpath to inspect specifically (e.g. 'src' or 'data')."},
            },
            "required": ["dirpath"],
        },
    },
    "file_grep": {
        "function": safe_grep_text,
        "description": (
            "Searches INSIDE files for code, function definitions (e.g. '_clean_final_answer'), variables, or phrases. "
            "Dynamically extracts the complete function body via AST and returns the exact file path and line number."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Function name, keyword, or exact text to find inside files."},
                "file_extension": {"type": "string", "description": "Optional file extension filter (e.g., '.py', '.js', '.txt')."},
            },
            "required": ["query"],
        },
    },
    "file_read": {
        "function": safe_read_file,
        "description": (
            "Reads text content from a specific file (.txt, .md, .py, .js, .json, .pdf, .docx) or inspects images/SVG. "
            "Accepts full paths or relative paths shown in dir_tree."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "filepath": {"type": "string", "description": "Full or relative path to the target file."},
                "start_line": {"type": "integer", "description": "Optional starting line number (1-indexed)."},
                "end_line": {"type": "integer", "description": "Optional ending line number."},
            },
            "required": ["filepath"],
        },
    },
}