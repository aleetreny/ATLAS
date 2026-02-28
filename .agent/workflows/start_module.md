---
description: Start a new ATLAS module with the correct Jupytext workflow and style matching.
---

I am ready to start the next section of the ATLAS project.

Please review the project structure: `instructions.md` and `README.md` to identify the next logical module we need to tackle.

**Setup Protocol:**

1.  **Identify**: Check `modules/` to see what is completed and confirm the next number/topic from the README.
2.  **Style Match**: Briefly read the _previous_ `.py` notebook in the sequence. Analyze its tone, structure (header depth), and visualization style ("premium" matplotlib/seaborn settings) to ensure absolute consistency.
3.  **Initialize**: Create the new **Editor File** as a `.py` file (e.g., `modules/.../0X_topic_name.py`) using the `percent` format.
4.  **Sync**: Immediately run `jupytext --to notebook filename.py` and `jupytext --sync filename.py` to generate the paired `.ipynb` Execution File.
5.  **Draft**: Write the **Complete Notebook Structure** in the `.py` file, incorporating all standard ATLAS phases:
    - **1. Intuition**: The physical/geometric "Why".
    - **2. Scratchpad**: The "from-scratch" NumPy implementation.
    - **3. Pro League**: The Industry Standard (Sklearn/etc) implementation.
    - **4. Visualization/Conclusion**: The "Decision Matrix" and final comparative plots.

**Start by confirming the next topic.**
