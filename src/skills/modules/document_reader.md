## Sub-Skill Module: Document & Data Reader (`file_read`)

### Tool Specifications & Parameter Rules
- **`file_read`**: Reads extracted text from documents (`.pdf`, `.docx`, `.txt`, `.md`) or structured data files (`.json`, `.js`, `.csv`).
  - `filepath` (`string`, required): Full path or relative path (from `dir_tree`) to the target file. Prefer paths inside `/Documents/` over duplicate `(1).pdf` copies in `/Downloads/` when both exist.
  - `start_line` / `end_line` (`integer`, optional): 1-indexed line range for partial reads of very large text files.
  - **Expected Output**: Returns `[File read: <path>]` followed by page-by-page extracted text (`--- Page X (of Y) ---`) and a `[Key Document Header & Identifiers Recap]` at the bottom for PDFs, or full text up to 20,000 characters.
  - **Accuracy Rule**: When extracting identifiers such as a `Matrikelnummer` / `Matriculation number`, always read the exact 7-digit number following `Matrikelnummer:` / `Matriculation number:` (never confuse it with a 4-digit calendar year like `2026`).

### Operational Workflow & Step-by-Step Example (Finding and Reading a PDF)
- **Step 1 (Search for the file if path is unknown)**:
  <thought>The user wants to find a specific PDF document, so I will call file_search filtering by entry_type file.</thought>
  {"action": "file_search", "arguments": {"pattern": "StudentAngemeldetePruefungde7c28e6-a6cf-45df-808b-0b40ebb4a23e.pdf", "entry_type": "file"}}
- **Step 2 (Read the file found)**:
  <thought>I have the exact path to the PDF from my file_search execution; now I will call file_read to extract its text.</thought>
  {"action": "file_read", "arguments": {"filepath": "/Users/juanfelenis/Downloads/StudentAngemeldetePruefungde7c28e6-a6cf-45df-808b-0b40ebb4a23e.pdf"}}
- **Step 3 (Write Final Answer once `file_read` output is present — NO JSON)**:
  <thought>I have read the complete PDF text; I will now answer in English including both the registered courses and the exact document date.</thought>
  The registered courses listed in the document are:
  - **Python vs Java**
  - **Artificial Intelligence**
  - **EdgeML meets America Football**

  The exact date on the document is **21. August 2026**.