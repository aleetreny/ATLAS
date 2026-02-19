# ATLAS Agent Instructions: Core Workflow & Principles

This document serves as the persistent memory for the AI agent working on the ATLAS repository. Follow these rules strictly to ensure consistency, quality, and structural integrity.

---

## Project Architecture

- `**modules/**`: Story-driven notebooks (Paired via Jupytext).
  - **The Jupytext Workflow**: For every analysis, maintain two files in the module folder:
    - `notebook.py`: The **Editor File**. The Agent edits this file (clean Python).
    - `notebook.ipynb`: The **Execution File**. The User runs this file.
  - **Why?**: This prevents JSON merge conflicts and allows the Agent to write cleaner code.
- `**src/models/`**: The "Scratch" Library.
  - Contains the *finalized*, reusable implementations of algorithms.
  - Move code here only after it has been proven to work in the notebook.
- `**src/utils/*`*: The utility belt.
  - `visuals.py`: House all "premium" plotting functions here to keep notebooks clean.
  - `loaders.py`: Helper scripts for fetching/cleaning data.
- `**datasets/**`: Keep it lean.
  - Store small CSVs (<10MB).
  - Use script-based loading for larger datasets to avoid GitHub bloat.

---

## The ATLAS Implementation Loop (MANDATORY)

For every new technique or case study, follow these four steps in order:

1. **Intuition**: Write a concise, clear explanation of the real-world problem and the mathematical concept. Use LaTeX for equations.
2. **Scratch**:
  - **Draft**: Implement the algorithm (using NumPy only) directly in the `notebook.py`.
  - **Refactor**: Once the implementation is solid, move the class to `src/models/` (e.g., `src/models/regularization.py`).
3. **Pro**: Implement the same technique using industry-standard libraries (Scikit-Learn, PyTorch, etc.).
4. **Visual**: Compare Scratch vs. Pro and analyze results using high-quality, interactive visualizations (Seaborn, Plotly).

---

## Creative & Technical Guidelines

- **Tone**: Professional, neutral, and clear. Avoid "grandiloquent" or unnecessary adjectives. Maintain a "geeky" yet practical vibe focused on real-world utility.
- **Visuals**: Plots must look "premium". Use custom themes, clear labels, and meaningful color palettes. No default Matplotlib settings.
- **Engineering**: Always prioritize reusability in `src/`. If a function might be useful in another module, it belongs in `src/utils/`.
- **Large Data**: Never commit large binary files. If a dataset is large, write a downloader script in `src/utils/loaders.py`.

---

## Critical Restrictions

- `**src/models/`** must NOT import `scikit-learn`, `tensorflow`, or `keras`. Only math/logic libraries (NumPy, SciPy where basic math is needed).
- **README.md** is the source of truth for the roadmap. Do not change its core structure without explicit instructions.
- **Instructions.md** is for internal agent context only (ignored by git).

