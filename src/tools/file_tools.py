"""Local read-only tools following the Single Responsibility Principle (SRP):
- file_search: locates files (entry_type='file') or folders (entry_type='dir') with extension filtering.
- dir_tree: compact directory tree exploration with relative paths, dense-folder summaries, and filters.
- file_grep: internal code/text search with dynamic complete function extraction via AST/braces.
- file_read: reads documents/code (supports absolute/relative paths and start_line..end_line), SVGs, and raster images.
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
    """Converts a string like '.js, .json, jsx' into {'.js', '.json', '.jsx'}."""
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
    exts = allowed_exts or set()
    want_dirs = mode in {"all", "dir", "directory", "folder"}
    want_files = (mode in {"all", "file", "files"}) or bool(exts)

    exact_matches: list[tuple[str, Path]] = []
    all_candidates: list[tuple[str, Path]] = []

    for root_dir in search_roots:
        for root, dirs, files in os.walk(root_dir, followlinks=False):
            dirs[:] = [d for d in dirs if not (d in IGNORE_DIR_NAMES or d.startswith("."))]

            if want_dirs:
                for d in dirs:
                    full_d = Path(root) / d
                    all_candidates.append(("DIR", full_d))
                    norm_d = _normalize(d)
                    if fnmatch.fnmatch(norm_d, glob_pattern) or (not has_wildcards and norm_pattern in norm_d):
                        exact_matches.append(("DIR", full_d))
                        if len(exact_matches) >= max_results:
                            return exact_matches

            if want_files:
                norm_root_str = _normalize(root)
                parent_matches = (not has_wildcards) and bool(norm_pattern) and (norm_pattern in norm_root_str)
                for file in files:
                    if file.startswith("."):
                        continue
                    full_f = Path(root) / file
                    if exts and full_f.suffix.lower() not in exts:
                        continue
                    all_candidates.append(("FILE", full_f))
                    norm_file = _normalize(file)
                    name_matches = fnmatch.fnmatch(norm_file, glob_pattern) or (
                        not has_wildcards and norm_pattern in norm_file
                    )
                    if name_matches or (exts and parent_matches):
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
    Resolves absolute paths, ~ paths, or paths relative to authorized workspaces or subprojects,
    tolerating '-' vs '_' differences in intermediate directory names.
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
    found_entries = _find_entries_by_name(target_name, _get_search_roots(), max_results=12, entry_type=entry_mode)

    def _norm_sep(s: str) -> str:
        return re.sub(r"[-_]+", "-", _normalize(s))

    norm_cleaned = _norm_sep(cleaned.lstrip("./"))
    matching_suffix = [
        path_obj for _, path_obj in found_entries
        if _norm_sep(str(path_obj)).endswith(norm_cleaned)
    ]
    candidates = matching_suffix if matching_suffix else [path_obj for _, path_obj in found_entries]
    exact_name = [c for c in candidates if _norm_sep(c.name) == _norm_sep(target_name)]
    chosen = exact_name if exact_name else candidates

    if len(chosen) > 1:
        input_parts = [_norm_sep(part) for part in p.parts if part not in {"/", ".", "~"} and part != p.name]
        if input_parts:
            scored_cands: list[tuple[int, Path]] = []
            for cand_path in chosen:
                cand_norm_str = _norm_sep(str(cand_path.parent))
                overlap = sum(1 for part in input_parts if part and part in cand_norm_str)
                scored_cands.append((overlap, cand_path))
            max_overlap = max((sc[0] for sc in scored_cands), default=0)
            if max_overlap > 0:
                best_cands = [cand_path for ov, cand_path in scored_cands if ov == max_overlap]
                if len(best_cands) == 1:
                    chosen = best_cands

    if len(chosen) == 1:
        resolved = chosen[0].resolve()
        note = f"[Auto-resolved path: {resolved}]\n"
        return resolved, [], note
    if len(chosen) > 1:
        return None, chosen, ""
    return p.resolve(), [], ""


def safe_dir_tree(
    dirpath: str,
    extension_filter: str = "",
    sub_dir: str = "",
    max_depth: int = 5,
    max_entries: int = 80,
) -> str:
    """
    Lists the directory tree using clean relative paths. Orders specific named files before generic
    numbered 'mystery_*' files so candidate inspection checks named files first.
    """
    try:
        target, multiples, auto_note = _resolve_flexible_path(dirpath, expect_dir=True)
        if multiples:
            opts = "\n".join(f"- {m}" for m in multiples)
            return f"Multiple directories matched '{dirpath}':\n{opts}\nPlease specify the desired path."
        if target is None or not target.exists():
            return f"Error: Directory '{dirpath}' does not exist."

        if sub_dir and sub_dir.strip():
            sub_target = (target / sub_dir.strip().strip("/")).resolve()
            if not sub_target.exists():
                return f"Error: Subfolder '{sub_dir}' does not exist inside '{target}'."
            target = sub_target

        if not _is_path_allowed(target):
            return f"Security Error: Access denied outside authorized workspaces ('{target}')."
        if not target.is_dir():
            return f"Error: '{target}' is a file, not a directory. Use `file_read` to inspect it."

        exts = _parse_extension_filter(extension_filter)
        depth_limit = max(1, min(int(max_depth or 5), 6))
        filter_info = f" | Filter: {sorted(exts)}" if exts else ""
        lines = [
            f"{auto_note}[Base Directory: {target}{filter_info}]",
            "(Paths relative to Base Directory — pass `'subfolder/filename'` or the full path to `file_read`):",
        ]

        count = 0
        per_folder_preview = 60 if (sub_dir and sub_dir.strip()) else 10

        for root, dirs, files in os.walk(target, followlinks=False):
            dirs[:] = sorted([d for d in dirs if not (d in IGNORE_DIR_NAMES or d.startswith("."))])
            rel_dir = Path(root).relative_to(target)
            depth = len(rel_dir.parts)
            if depth > depth_limit:
                dirs[:] = []
                continue

            visible_files = []
            # Sort specific semantic filenames first, generic 'mystery_*' / numbered filenames second
            sorted_files = sorted(
                files,
                key=lambda fname: (1 if fname.lower().startswith("mystery") else 0, fname.lower()),
            )
            for f in sorted_files:
                if f.startswith("."):
                    continue
                if exts and Path(f).suffix.lower() not in exts:
                    continue
                visible_files.append(f)

            indent = "  " * depth
            if depth > 0 and (visible_files or dirs):
                lines.append(f"{indent}📁 {rel_dir.as_posix()}/")

            shown_in_folder = visible_files[:per_folder_preview]
            hidden_in_folder = visible_files[per_folder_preview:]

            for f in shown_in_folder:
                rel_file = (rel_dir / f).as_posix() if depth > 0 else f
                lines.append(f"{indent}  📄 {rel_file}")
                count += 1
                if count >= max_entries:
                    lines.append(f"{indent}  ... [Reached {max_entries} entries limit. Use `sub_dir` or `extension_filter` to inspect a specific section]")
                    return "\n".join(lines)

            if hidden_in_folder:
                ext_counts = Counter(Path(f).suffix.lower() or "no_ext" for f in hidden_in_folder)
                ext_summary = ", ".join(f"{k} ({v})" for k, v in ext_counts.most_common(4))
                folder_label = rel_dir.as_posix() if depth > 0 else "."
                lines.append(
                    f"{indent}  ... (+{len(hidden_in_folder)} additional files in '{folder_label}': {ext_summary} "
                    f"— use sub_dir=\"{folder_label}\" to list all)"
                )

        if count == 0:
            if exts:
                lines.append(f"  (No files matching extensions {sorted(exts)} found in this directory)")
            else:
                lines.append("  (Empty directory)")
        return "\n".join(lines)

    except Exception as exc:
        return f"Error exploring directory '{dirpath}': {exc}"


def _read_pdf(path: Path, max_chars: int) -> str:
    reader = PdfReader(str(path))
    pages_text = []
    first_page_key_lines: list[str] = []
    total_len = 0
    for i, page in enumerate(reader.pages):
        text = _sanitize_text(page.extract_text() or "")
        if i == 0:
            non_empty = [ln.strip() for ln in text.splitlines() if ln.strip()]
            first_page_key_lines = non_empty[:8]
        pages_text.append(f"--- Page {i + 1} (of {len(reader.pages)}) ---\n{text}")
        total_len += len(text)
        if total_len >= max_chars:
            break
    full_text = "\n\n".join(pages_text)[:max_chars]

    # Append a concise header recap at the bottom of the PDF output so Gemma 4's 512-token
    # sliding-window attention layers always retain the exact top identifiers (e.g. Matrikelnummer & dates)
    if first_page_key_lines:
        recap_block = (
            "\n\n[Key Document Header & Identifiers Recap (from Top of Page 1)]:\n"
            + "\n".join(f"• {ln}" for ln in first_page_key_lines)
        )
        full_text += recap_block
    return full_text


def _read_docx(path: Path, max_chars: int) -> str:
    doc = Document(str(path))
    paragraphs = [_sanitize_text(p.text) for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)[:max_chars]


def _inspect_image(path: Path) -> str:
    """Extracts complete technical metadata and embeds the visual pipeline path marker."""
    with Image.open(path) as img:
        width, height = img.size
        img_format = (img.format or path.suffix.upper().replace(".", "")).upper()
        mode = img.mode
        size_kb = path.stat().st_size / 1024
        aspect = round(width / max(1, height), 2)
        resolved_path = str(path.resolve())

        return (
            f"[Image File Inspected]\n"
            f"- Format: {img_format} | Color Space / Mode: {mode}\n"
            f"- Dimensions: {width}x{height} px (Aspect Ratio: {aspect}:1)\n"
            f"- File Size: {size_kb:.1f} KB\n"
            f"- Absolute Path: {resolved_path}\n"
            f"[IMAGE_READY_FOR_VISUALIZATION: {resolved_path}]\n"
            f"Note: The raw pixels of this image are now attached directly to your visual input. "
            f"Inspect the attached image visually right now to describe the scene/UI, read any text or "
            f"secret code written inside the pixels, and verify whether the visual content matches the filename."
        )


def _inspect_svg(path: Path) -> str:
    """Extracts XML vector metadata and embedded text labels from an SVG file."""
    tree = ET.parse(path)
    root = tree.getroot()
    width = root.attrib.get("width", "unspecified")
    height = root.attrib.get("height", "unspecified")
    view_box = root.attrib.get("viewBox", "unspecified")
    size_kb = path.stat().st_size / 1024
    resolved_path = str(path.resolve())

    tag_counts: Counter[str] = Counter()
    text_labels: list[str] = []
    for elem in root.iter():
        tag_name = elem.tag.split("}")[-1] if "}" in elem.tag else elem.tag
        if tag_name != "svg":
            tag_counts[tag_name] += 1
        if tag_name in {"text", "title", "desc"} and elem.text and elem.text.strip():
            text_labels.append(elem.text.strip())

    elements_summary = ", ".join(f"{k}: {v}" for k, v in tag_counts.most_common(6)) or "none"
    labels_line = f"- Embedded Text/Titles: {', '.join(text_labels[:5])}\n" if text_labels else ""

    return (
        f"[SVG Vector File Inspected]\n"
        f"- Format: SVG (XML Vector Graphics)\n"
        f"- Dimensions: width={width}, height={height} | viewBox={view_box}\n"
        f"- File Size: {size_kb:.1f} KB\n"
        f"- Vector Elements Count: {elements_summary}\n"
        f"{labels_line}"
        f"- Absolute Path: {resolved_path}\n"
        f"Note: SVG files are XML vector definitions (not raster bitmaps). Use these technical metadata to answer."
    )


def safe_read_file(
    filepath: str,
    start_line: int | None = None,
    end_line: int | None = None,
    max_chars: int = 20_000,
) -> str:
    """Reads file contents supporting absolute or relative paths and optional line ranges."""
    try:
        raw_clean = filepath.strip().strip("'\"")
        if Path(raw_clean).name.startswith("."):
            return f"Security Error: Access denied to hidden item ('{raw_clean}')."

        target, multiples, auto_discovered_note = _resolve_flexible_path(raw_clean, expect_dir=False)
        if multiples:
            paths_str = "\n".join(f"- {p}" for p in multiples)
            return (
                f"Multiple files matched '{filepath}':\n{paths_str}\n"
                "Please call `file_read` specifying the full path you want to read."
            )

        if target is None or not target.exists():
            return f"Error: File '{filepath}' was not found in authorized workspaces."

        if not _is_path_allowed(target):
            return f"Security Error: Access denied outside authorized workspaces ('{filepath}')."

        target_name_lower = target.name.lower()
        if any(blocked in target_name_lower for blocked in BLOCKED_PATTERNS):
            return f"Security Error: Access denied to protected item ('{target.name}')."

        if target.is_dir():
            return (
                f"[Note: '{target.name}' is a directory; listing its structure via `dir_tree`]\n"
                + safe_dir_tree(str(target))
            )

        suffix = target.suffix.lower()
        if suffix in BLOCKED_EXTENSIONS:
            return f"Security Error: Access denied to certificate or key file ('{suffix}')."

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
                    f"Line {i:4d} | {all_lines[i - 1]}"
                    for i in range(s_idx, e_idx + 1)
                ]
                content = (
                    f"[Showing lines {s_idx} to {e_idx} of {total_lines} total]\n"
                    + "\n".join(sliced)
                )
            else:
                content = raw_text

            if len(content) > max_chars:
                clipped_part = content[:max_chars]
                lines_shown = clipped_part.count("\n") + 1
                content = (
                    clipped_part
                    + f"\n... [File truncated at line {lines_shown} of {total_lines} total ({max_chars} of {len(content)} chars). "
                    f"Use `start_line={lines_shown + 1}` and `end_line={total_lines}` in `file_read` to read the remaining lines]."
                )

        header = f"[File read: {target}]\n"
        return auto_discovered_note + header + content

    except Exception as exc:
        return f"Error reading '{filepath}': {exc}"


def safe_search_files(
    pattern: str,
    entry_type: str = "all",
    extension_filter: str = "",
    search_dir: str = "",
    max_results: int = 20,
) -> str:
    """Locates files ([FILE]) and/or directories ([DIR]) by exact or fuzzy name."""
    try:
        base_path = Path(search_dir).expanduser().resolve() if search_dir else None
        if base_path and not _is_path_allowed(base_path):
            return f"Security Error: Directory '{search_dir}' is not permitted."

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
            res = [f"Found items ({len(matches)}):"]
            for kind, p in matches:
                res.append(f"- [{kind}] {p}")
            return "\n".join(res)

        return (
            f"No items found for pattern='{pattern}' (entry_type='{entry_type}', extension_filter='{extension_filter}'). "
            "Try a broader single keyword or adjust the filters."
        )

    except Exception as exc:
        return f"Error searching files: {exc}"


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
            return f"Security Error: Directory '{search_dir}' is not authorized."

        targets = [base_path] if (base_path and base_path.exists()) else _get_search_roots()
        def_findings: list[str] = []
        other_findings: list[str] = []
        norm_query = _normalize(query.strip().strip("'\""))
        exts = _parse_extension_filter(file_extension)
        ctx_window = max(3, min(int(context_lines or 12), 80))

        if not norm_query:
            return "Error: Search query is empty."

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

                                tag = f"[COMPLETE DEFINITION: lines {start_c + 1} to {end_c}]" if is_definition else "[REFERENCE]"
                                block_lines = [
                                    f"File: {file_path} | Exact match on LINE {line_num} {tag}:"
                                ]
                                for c_idx in range(start_c, end_c):
                                    code_line = lines[c_idx][:200]
                                    block_lines.append(f"Line {c_idx + 1:4d} | {code_line}")

                                if is_definition:
                                    block_lines.append(
                                        f"[AST Definition Bounds Recap: File={file_path} | "
                                        f"Exact Definition Line={start_c + 1} (complete range: lines {start_c + 1} to {end_c})]"
                                    )

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
            return f"No matches found for '{query}' across files."

        header = f"Found matches ({len(combined)} blocks):"
        return header + "\n\n" + "\n\n".join(combined)

    except Exception as exc:
        return f"Error searching text inside files: {exc}"


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
                "pattern": {"type": "string", "description": "Single keyword or filename pattern (e.g., 'portfolio', 'Immatrikulationsbescheinigung')."},
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
            "Searches INSIDE files for code, function definitions (e.g. '_compose_modular_skill'), variables, or phrases. "
            "Dynamically extracts the complete function body via AST and returns the exact file path and line numbers."
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
            "Reads text content from a specific file (.txt, .md, .py, .js, .json, .pdf, .docx), inspects SVG metadata, "
            "or loads raster images (.png, .jpg, .jpeg, .webp) into the multimodal Computer Vision pipeline alongside technical metadata."
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