# hetmacro — References

## Core algorithmic references

- **CompEcon Toolbox** (Miranda & Fackler) — quadrature, basis functions, Broyden solver. hetmacro reimplements key routines in NumPy.
- **Auclert et al. (2021)** "Using the Sequence-Space Jacobian to Solve and Estimate Heterogeneous-Agent Models" — basis for `ssj.py`.
- **Rouwenhorst (1995)** — income discretization method in `markov.py`.
- **Tauchen (1986)** — alternative income discretization in `markov.py`.
- **Carroll (2006)** "The Method of Endogenous Gridpoints for Solving Dynamic Stochastic Optimization Problems" — basis for EGM solver.

## Course context

- **Virgiliu Midrigan's Heterogeneity in Macro** — problem sets 1-3 drive the examples.

## Documentation

- `docs/codebook/codebook.pdf` — canonical API reference (rebuild with `latexmk -pdf codebook.tex`).
- `docs/compecon_comparison.md` — notes on differences from CompEcon.
- `macro_agents.md` — AI agent playbooks for using the toolkit.
