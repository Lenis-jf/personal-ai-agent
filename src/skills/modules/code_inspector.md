## Sub-Skill Module: Source Code Inspector & Formatter (`file_read` & Code Display)

### Reading Source Code Files (`file_read`)
- Use `file_read` when you want to inspect a known source code file (`.py`, `.js`, `.jsx`, `.ts`, `.java`, `.c`, `.cpp`, `.scss`) either completely or across a specific line range (`start_line`, `end_line`).
- If you are looking for a specific function or class definition across the repository, prefer `file_grep` first, as it locates the file and extracts the complete function body with `Línea XXXX |` prefixes automatically.

### Operational Workflow & Step-by-Step Example (Displaying & Explaining Code)
- **Step 1 (After obtaining the code from `file_grep` or `file_read` — NO CODE FENCES)**:
  Never wrap source code in triple backticks (` ``` `). Always display the exact metadata header, dashed separators, and line-numbered code, followed by your technical explanation:
  <thought>Ya tengo el bloque completo de la función con sus números de línea exactos; ahora lo presentaré con el formato plano obligatorio y explicaré su lógica.</thought>
  - **Lenguaje**: Python
  - **Archivo**: /Users/juanfelenis/Documents/Proyectos/personal-agent/src/agent.py
  - **Línea exacta de definición**: Línea 317 (extensión completa: líneas 317 a 339)
  ----------------------------------------
  Línea  317 |     def _split_leading_thought(cls, raw_text: str) -> tuple[str, str]:
  Línea  318 |         ...
  ----------------------------------------

  **Explicación técnica:**
  - **Método / Expresión**: Detalle claro de qué hace cada parte y por qué.