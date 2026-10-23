## Sub-Skill Module: Document & Data Reader (`file_read`)

### Tool Specifications & Parameter Rules
- **`file_read`**: Reads extracted text from documents (`.pdf`, `.docx`, `.txt`, `.md`) or structured data files (`.json`, `.js`, `.csv`).
  - `filepath` (`string`, required): Full path or relative path (from `dir_tree`) to the target file.
  - `start_line` / `end_line` (`integer`, optional): 1-indexed line range for partial reads of very large text files.
  - **Expected Output**: Returns `[Archivo leído: <path>]` followed by page-by-page extracted text (`--- Página X (de Y) ---`) for PDFs or full text up to 20,000 characters. If a file exceeds 20,000 characters, it appends a truncation notice with the exact line number so you can call `file_read` with `start_line` and `end_line` if needed.

### Operational Workflow & Step-by-Step Example (Finding and Reading a PDF)
- **Step 1 (Search for the file if path is unknown)**:
  <thought>El usuario pide buscar un archivo PDF específico, así que usaré file_search filtrando por entry_type file.</thought>
  {"action": "file_search", "arguments": {"pattern": "StudentAngemeldetePruefungde7c28e6-a6cf-45df-808b-0b40ebb4a23e.pdf", "entry_type": "file"}}
- **Step 2 (Read the file found)**:
  <thought>Ya tengo la ruta exacta del PDF en Downloads; ahora usaré file_read para extraer su texto.</thought>
  {"action": "file_read", "arguments": {"filepath": "/Users/juanfelenis/Downloads/StudentAngemeldetePruefungde7c28e6-a6cf-45df-808b-0b40ebb4a23e.pdf"}}
- **Step 3 (Write Final Answer once `file_read` output is present — NO JSON)**:
  <thought>Ya leí el contenido completo del PDF; responderé incluyendo tanto la lista de materias como la fecha exacta escrita en el documento.</thought>
  Las materias inscritas en el documento son:
  - **Python vs Java**
  - **Artificial Intelligence**
  - **EdgeML meets America Football**

  La fecha exacta del documento es **21. August 2026**.