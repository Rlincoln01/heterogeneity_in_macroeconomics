# Examples

Example uses of **hetmacro**.

- **pset1/** - Problem set 1 notebooks (canonical source: course "My code and solutions/notebooks").
  - Topics: quadrature, nonlinear solvers (Broyden, Nelder-Mead), two-period labor supply, portfolio choice.
  - Location: `examples/pset1/notebooks/`.

- **pset2/** - Problem set 2 notebooks and slides (canonical source: course "My Code and Solutions").
  - Topics: Rouwenhorst discretization diagnostics, cake-eating with collocation/projection methods, lifecycle model.
  - Locations:
    - `examples/pset2/notebooks/`
    - `examples/pset2/slides/`
    - `examples/pset2/python/` (helper module used by problem 4 notebook)

- **pset3/** - Problem set 3 notebooks (canonical source: course "My Code and Solutions/notebooks_v2", excluding backup files).
  - Topics: income fluctuation problems (discrete and AR(1) income), VFI, Howard improvement, ergodic distributions.
  - Location: `examples/pset3/notebooks/`.

## Notes

- Notebooks are configured to auto-detect the repository root by searching parent directories for `hetmacro/`.
- This avoids machine-specific absolute paths and keeps imports stable after relocations.
