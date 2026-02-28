---
description: Extract the Scratch class from the current notebook and save it to the src library.
---

I want to immortalize the "Scratch" implementation from this notebook into our core library.

**Protocol:**

1.  **identify**: Locate the `class ScratchName` definition in the current open notebook (`.py` file).
2.  **Locate Destination**: Determine the correct file in `src/models/` based on the notebook's topic (e.g., `src/models/regression.py` or a new file if the topic implies a new category).
3.  **Transfer**:
    - Read the destination file to checks if the class already exists.
    - If not, **Append** the full class definition (including all helper methods) to the end of the destination file.
    - Ensure the file has necessary imports (e.g., `import numpy as np`) at the top.
4.  **Verify**: Check that `src/models/__init__.py` exists.

**Constraint**: Do **NOT** remove the class from the notebook. The notebook must remain self-contained for educational purposes. Just copy it to the library.
