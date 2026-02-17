# Codebook folder

- **Source**: `codebook.tex` — edit this file to change the codebook; then rebuild the PDF.
- **Output**: `codebook.pdf` — compiled narrative codebook for hetmacro.

Rebuild from repo root or this directory, e.g.:
```bash
latexmk -pdf codebook.tex
```

If the PDF is out of date with the Python API, update the relevant sections in `codebook.tex` to match the current function signatures in `hetmacro/*.py`.

## GitHub sync rule (project policy)

If you change public functions in `hetmacro/*.py` **or** update `codebook.tex`, those changes must be committed and pushed to GitHub (via the dev branch workflow in `.cursor/rules/20-git-autopush.mdc`) so the repository documentation always matches the latest package code.
