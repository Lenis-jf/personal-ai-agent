## Sub-Skill Module: Computer Vision & Vector Image Inspection (`file_read`)

### Tool Behavior on Visual & Vector Assets
- **Raster Images (`.png`, `.jpg`, `.jpeg`, `.webp`, `.bmp`, `.gif`, `.tiff`)**:
  Calling `file_read` on a raster image performs TWO actions simultaneously:
  1. Extracts an English technical metadata card: `Format`, `Color Space / Mode` (`RGB`, `RGBA`), `Dimensions` (`width x height px`), `Aspect Ratio` (`X:1`), `File Size` (`KB`), and `Absolute Path`.
  2. Injects the actual image pixels directly into your native Gemma 4 multimodal visual input so you can visually inspect the scene, UI components, colors, and read embedded text (OCR) in the very next step.
- **Vector Graphics (`.svg`)**:
  Calling `file_read` on an `.svg` file parses its XML tree and returns its `Dimensions`, `viewBox`, `File Size`, `Vector Elements Count` (`rect`, `circle`, `path`, `text`), and `Embedded Text/Titles`. Since SVGs are XML vector definitions rather than raster bitmaps, answer directly using these extracted metadata.

### Operational Workflows & Examples for Images
1. **Case A — Location Only**:
   If the user ONLY asks where an image file is located, use `file_search` or `dir_tree` and report the path directly without calling `file_read`.
2. **Case B — Inspecting a Specific Image (Visual Content + Technical Metadata)**:
   - **Step 1**: Locate the image with `file_search` (e.g., `{"pattern": "dronesim", "extension_filter": ".png"}`) or `dir_tree`.
   - **Step 2**: Call `file_read` on the exact image path returned by step 1:
     <thought>I found dronesim-img-1.png; now I will call file_read to load its pixels into the Computer Vision pipeline and extract its technical metadata.</thought>
     {"action": "file_read", "arguments": {"filepath": "/Users/juanfelenis/Documents/Proyectos/portfolio-2.0-vite/public/assets/imgs/dronesim-assets/dronesim-img-1.png"}}
   - **Step 3**: In the next step, examine the attached image visually and write your complete natural language description combining what you see in the interface/scene with its resolution and aspect ratio.
3. **Case C — Visual Search by Content & Ground Truth Verification (Overcoming Textual Confirmation Bias)**:
   When the user asks you to find which image in a folder shows a specific visual subject (e.g., "a yellow duck in the water"):
   - **Rule 1 (Candidate Gathering)**: When the user specifies a folder name (e.g., `test_sandbox`), ALWAYS use `dir_tree` (e.g., `{"dirpath": "test_sandbox", "extension_filter": ".png"}`) to list all candidate images inside that folder.
   - **Rule 2 (Inspect Named/Suspect Decoy Files FIRST)**: Always start by inspecting the first file listed by `dir_tree`—especially any file whose name claims to be the target object in Spanish or English (such as `pato.png` = duck) to verify via Computer Vision whether it is genuine or a deceptive decoy trap! Never assume a file matches based solely on its filename.
   - **Rule 3 (Adversarial / Deceptive Name Recovery)**: If `file_read` on `pato.png` reveals visually that it shows something else (e.g., a red sports car with a trap message instead of a duck), do NOT stop:
     <thought>I inspected pato.png and visually confirmed it is a deceptive decoy showing a red sports car instead of a yellow duck. I will discard pato.png and call file_read on the next candidate mystery_img_01.png.</thought>
     {"action": "file_read", "arguments": {"filepath": "test_sandbox/mystery_img_01.png"}}
   - **Rule 4 (Iterative Inspection & Transparent Reporting)**: Continue calling `file_read` on the remaining candidates (`mystery_img_01.png`, `mystery_img_02.png`) until you visually confirm the real target. In your final answer, report all deceptive files you discarded (`pato.png`, `mystery_img_01.png`) as well as the real matching image, its secret code, and its exact dimensions.