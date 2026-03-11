# hetmacro — Runbook

## Setup

```bash
cd "Heterogeneity in Macro/Heterogeneity In Macroeconomics - Github"
pip install -r requirements.txt
# Python 3.11+ recommended
```

## Running examples

```bash
# Package test notebooks
jupyter notebook hetmacro/notebooks/

# Problem set notebooks
jupyter notebook examples/pset1/notebooks/
jupyter notebook examples/pset2/notebooks/
jupyter notebook examples/pset3/notebooks/

# Benchmark script
python examples/benchmark_howard_euler_speed.py
```

## Running tests

```bash
pytest tests/
```

## Building codebook

```bash
cd docs/codebook
latexmk -pdf codebook.tex
```

## Building transition path methods doc

```bash
cd docs
pdflatex transition_path_methods.tex && pdflatex transition_path_methods.tex
```

## Git workflow

```bash
# Always work on dev branch
git checkout dev/hetmacro-sync

# After changes: stage, commit, push
git add <files>
git commit -m "hetmacro: <why-focused message>"
git push origin dev/hetmacro-sync

# NEVER commit on main. NEVER force-push.
# If API changed, update codebook.tex and rebuild PDF before committing.
```

## Remote

```
origin  git@github.com:Rlincoln01/heterogeneity_in_macroeconomics.git
```

## Branch status (as of 2026-03-11)

- `dev/hetmacro-sync`: ahead of `main`. Uncommitted work being pushed this session.
