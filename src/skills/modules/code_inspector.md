## Sub-Skill Module: Source Code Inspector & Formatter (`file_read` & Code Display)

### Reading Source Code Files (`file_read`)
- Use `file_read` when you want to inspect a known source code file (`.py`, `.js`, `.jsx`, `.ts`, `.java`, `.c`, `.cpp`, `.scss`) either completely or across a specific line range (`start_line`, `end_line`).
- If you are looking for a specific function or class definition across the repository, prefer `file_grep` first, as it locates the file and extracts the complete function body with `Line XXXX |` prefixes automatically.

### Operational Workflow & Step-by-Step Example (Displaying & Explaining Code)
- **Step 1 (After obtaining the code from `file_grep` or `file_read` — NO CODE FENCES)**:
  Never wrap source code in triple backticks (` ``` `). Always display the exact metadata header, dashed separators, and every single line-numbered code line verbatim from start to end, followed by your technical explanation:
  <thought>I now have the complete AST-extracted function block with its exact line numbers; I will display the complete verbatim code using plain dashed separators and explain its logic in English.</thought>
  **Language**: Python
  **File**: /Users/juanfelenis/Documents/Proyectos/personal-agent/src/agent.py
  **Exact Line(s)**: Line 321 (complete definition: lines 321 to 382)
  ----------------------------------------
  Line  321 |     def _compose_modular_skill(
  Line  322 |         self,
  ...
  Line  382 |         return "\n\n".join(sections), " + ".join(loaded_labels)
  ----------------------------------------

  **Technical Explanation:**
  - **Base Skill Selection**: Explains how the primary skill file is resolved and loaded.
  - **Sub-Skill Module Assembly**: Explains how `MODULE_ORDER` and `decision.sub_skills` are combined.