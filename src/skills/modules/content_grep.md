## Sub-Skill Module: Internal Content & AST Code Search (`file_grep`)

### Tool Specifications & Parameter Rules
- **`file_grep`**: Searches INSIDE text and code files for exact text lines, function definitions, classes, variables, or phrases.
  - `query` (`string`, required): Exact function name, class name, variable identifier, or text phrase to find inside files (e.g., `"_split_leading_thought"`, `"decide_route"`).
  - `file_extension` (`string`, optional): Restrict scan to specific file extensions (e.g., `".py"`, `".js,.jsx"`, `".md"`).
  - **Expected Output**: Returns matching blocks with the header `Archivo: <full_path> | Coincidencia exacta en LÍNEA X [DEFINICIÓN COMPLETA: líneas X a Y]:` followed by the complete AST-extracted function body with `Línea XXXX | <code_line>`.
  - **Critical Constraint**: Once `file_grep` returns `[DEFINICIÓN COMPLETA: líneas X a Y]`, you ALREADY have the entire function code and line numbers—do NOT call `file_read` on the same file unless you need lines outside that function.

### Step-by-Step Example
- **Step 1 (Invoke `file_grep`)**:
  <thought>Voy a buscar en qué archivo Python está definida la función _split_leading_thought usando file_grep.</thought>
  {"action": "file_grep", "arguments": {"query": "_split_leading_thought", "file_extension": ".py"}}