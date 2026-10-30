## Sub-Skill Module: File & Directory Locator (`file_search` & `dir_tree`)

### Tool Specifications & Parameter Rules
1. **`file_search`**: Locates files (`[FILE]`) or directories (`[DIR]`) across all authorized workspaces.
   - `pattern` (`string`, required): Single main keyword or filename pattern (e.g., `"Immatrikulationsbescheinigung"`, `"portfolio"`, `"dronesim"`). **Constraint**: Never combine multiple unrelated words in `pattern`.
   - `entry_type` (`string`, optional): Set to `"dir"` when looking for a project folder, `"file"` when looking for a document/image, or `"all"`.
   - `extension_filter` (`string`, optional): Filter by file extension (e.g., `".pdf"`, `".docx"`, `".png,.jpg,.webp"`). Do not guess extensions if unknown.
   - **Expected Output**: Returns a list of matching paths prefixed with `- [FILE] /full/path` or `- [DIR] /full/path`.
2. **`dir_tree`**: Explores and lists the internal structure of a directory (`[DIR]`) using clean relative paths.
   - `dirpath` (`string`, required): Full path or project folder name to explore (e.g., `"/Users/juanfelenis/Documents/Proyectos/portfolio-2.0-vite"` or `"test_sandbox"`).
   - `extension_filter` (`string`, optional): Comma-separated extensions to filter when looking for specific file types (e.g., `".js,.json,.jsx"` for project data/code, or `".png,.jpg,.webp,.svg"` for images/assets). Omit to see all types.
   - `sub_dir` (`string`, optional): Specific subfolder inside `dirpath` to inspect in full detail (e.g., `"public/assets/imgs"`).
   - **Expected Output**: Returns `[Base Directory: <path>]` followed by folders (`📁`) and relative file paths (`📄`).

### Operational Workflow & Step-by-Step Example (Exploring a Project Folder)
- **Step 1 (Find the project directory)**:
  <thought>The user wants to explore their web portfolio directory, so I will search for folders matching the keyword portfolio.</thought>
  {"action": "file_search", "arguments": {"pattern": "portfolio", "entry_type": "dir"}}
- **Step 2 (Explore the directory structure)**:
  <thought>I found the portfolio-2.0-vite directory; now I will use dir_tree filtering by data and code files (.js, .json) to locate the projects data file.</thought>
  {"action": "dir_tree", "arguments": {"dirpath": "/Users/juanfelenis/Documents/Proyectos/portfolio-2.0-vite", "extension_filter": ".js,.json"}}
- **Step 3 (Read the target file discovered in the tree)**:
  <thought>The directory tree shows data/projectsData.js, so I will read it with file_read to extract the project details.</thought>
  {"action": "file_read", "arguments": {"filepath": "/Users/juanfelenis/Documents/Proyectos/portfolio-2.0-vite/data/projectsData.js"}}