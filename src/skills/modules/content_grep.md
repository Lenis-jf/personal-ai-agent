## Sub-Skill Module: Internal Content & AST Code Search (`file_grep`)

### Tool Specifications & Parameter Rules
- **`file_grep`**: Searches INSIDE text and code files for exact text lines, function definitions, classes, variables, or phrases.
  - `query` (`string`, required): Exact function name, class name, variable identifier, or text phrase to find inside files (e.g., `"_compose_modular_skill"`, `"decide_route"`).
  - `file_extension` (`string`, optional): Restrict scan to specific file extensions (e.g., `".py"`, `".js,.jsx"`, `".md"`).
  - **Expected Output**: Returns matching blocks with the header `File: <full_path> | Exact match on LINE X [COMPLETE DEFINITION: lines X to Y]:` followed by the complete AST-extracted function body with `Line XXXX | <code_line>`, and an `[AST Definition Bounds Recap]` at the bottom.
  - **Critical Constraint**: Once `file_grep` returns `[COMPLETE DEFINITION: lines X to Y]`, you ALREADY have the entire function code and exact line numbers—do NOT call `file_read` on the same file, and copy all lines from `Line X` to `Line Y` verbatim without inventing or skipping lines.

### Step-by-Step Example
- **Step 1 (Invoke `file_grep`)**:
  <thought>I will search across Python files for the definition of _compose_modular_skill using file_grep.</thought>
  {"action": "file_grep", "arguments": {"query": "_compose_modular_skill", "file_extension": ".py"}}