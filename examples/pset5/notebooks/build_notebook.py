"""Generate pset5_aiyagari_ge.ipynb programmatically."""
import json, pathlib

cells = []

def md(src):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": src.strip().splitlines(True)})

def code(src):
    cells.append({"cell_type": "code", "metadata": {}, "source": src.strip().splitlines(True), "outputs": [], "execution_count": None})

# ── 1. Title ────────────────────────────────────────────────────────────
md("""
# PSet 5: Aiyagari GE Steady State

Solve the Aiyagari (1994) model in general equilibrium using the `hetmacro` package.

**Outline:**
1. Setup and parameterization (PS5 calibration)
2. Capital supply and demand curves
3. GE steady state: Brent vs bisection root finding
4. Multi-solver comparison (EGM, GridVFI, PFI, HowardGrid, HowardImprovement, CollocationVFI_Spline)
5. Diagnostics: policy functions, wealth distribution, Euler residuals
""")

# ── 2. Imports ──────────────────────────────────────────────────────────
code("""
import importlib
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Repo-root detection
REPO_ROOT = None
for p in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
    if (p / "hetmacro").exists():
        REPO_ROOT = p
        break
if REPO_ROOT is None:
    raise RuntimeError("Could not locate repository root containing 'hetmacro'.")
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hetmacro.models.aiyagari
import hetmacro.household
import hetmacro.solvers.egm
import hetmacro.solvers.grid_vfi
import hetmacro.solvers.policy_iteration
import hetmacro.solvers.howard_grid
import hetmacro.solvers.howard
import hetmacro.solvers.collocation_vfi
importlib.reload(hetmacro.models.aiyagari)
importlib.reload(hetmacro.household)
importlib.reload(hetmacro.solvers.egm)
importlib.reload(hetmacro.solvers.grid_vfi)
importlib.reload(hetmacro.solvers.policy_iteration)
importlib.reload(hetmacro.solvers.howard_grid)
importlib.reload(hetmacro.solvers.howard)
importlib.reload(hetmacro.solvers.collocation_vfi)

from hetmacro.grids import make_asset_grid
from hetmacro.income_process import RouwenhorstIncome
from hetmacro.household import Household, SolvedPolicy
from hetmacro.markov import stationary_distribution
from hetmacro.models.aiyagari import (
    firm_k_demand, firm_wage,
    capital_supply_curve, solve_aiyagari_ge,
)
from hetmacro.solvers import (
    EGM, GridVFI, PolicyFunctionIteration,
    HowardGrid, HowardImprovement,
    CollocationVFI_Spline,
)

print("Imports OK")
""")

# ── 3. Parameters & Setup ──────────────────────────────────────────────
md("""
## 1. Parameters and Setup

PS5 calibration: $\\beta = 0.99$, $\\theta = 1.5$ (CRRA), $\\rho = 0.99$, $\\sigma_e = 0.10$,
$\\alpha = 0.35$, $\\delta = 0.02$. Asset grid: 501 points on $[0, 2500]$, power-spaced.
Income: Rouwenhorst with 11 states, normalized so $E[e] = 1$.
""")

code("""
# PS5 parameterization
beta = 0.99
gamma = 1.5          # CRRA (theta in PS5)
rho = 0.99
sigma_e = 0.10
alpha = 0.35
delta = 0.02

# Grid
nk = 501
ne = 11
k_min, k_max = 0.0, 2500.0

a_grid = make_asset_grid(k_min, k_max, nk, curvature=2.0)

# Income process (Rouwenhorst)
income = RouwenhorstIncome.from_ar1(n=ne, rho=rho, sigma=sigma_e)

# Normalize z_grid so E[z] = 1 under stationary distribution
pi_stat = stationary_distribution(income.Pi, method="iterate")
mean_z = float(np.dot(pi_stat, income.z_grid))
income.z_grid = income.z_grid / mean_z

print(f"Asset grid: {nk} points on [{k_min}, {k_max}]")
print(f"Income states: {ne}")
print(f"z_grid range: [{income.z_grid.min():.4f}, {income.z_grid.max():.4f}]")
print(f"E[z] under stationary dist: {np.dot(pi_stat, income.z_grid):.6f}")

# Interest rate bounds: r must be below 1/beta - 1 for convergence
r_upper = 1.0 / beta - 1.0
print(f"\\n1/beta - 1 = {r_upper:.6f}")
r_bounds = (-0.01, r_upper - 1e-4)
print(f"Search bounds: {r_bounds}")
""")

# ── 4. Supply/Demand Curves ────────────────────────────────────────────
md("""
## 2. Capital Supply and Demand Curves

The demand curve $K_d(r) = (\\alpha / (r + \\delta))^{1/(1-\\alpha)}$ is analytical and
downward-sloping. The supply curve $K_s(r)$ is the aggregate asset holdings
from the household problem at each interest rate, and is upward-sloping.
""")

code("""
# Compute supply curve on a grid of interest rates
print("Computing capital supply curve (this may take a few minutes)...")
curves = capital_supply_curve(
    solver=EGM(tol=1e-8, max_iter=2000),
    income_process=income,
    a_grid=a_grid,
    beta=beta,
    gamma=gamma,
    alpha=alpha,
    delta=delta,
    n_points=15,
    r_bounds=r_bounds,
    verbose=True,
)
print("Done.")
""")

code("""
# Plot supply vs demand
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(curves["K_demand"], curves["r"], "b-", linewidth=2, label="$K_d(r)$ (demand)")
ax.plot(curves["K_supply"], curves["r"], "r-o", linewidth=2, markersize=4, label="$K_s(r)$ (supply)")
ax.set_xlabel("Capital $K$", fontsize=12)
ax.set_ylabel("Net interest rate $r$", fontsize=12)
ax.set_title("Aiyagari Steady State: Capital Supply and Demand", fontsize=13)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
plt.tight_layout()
plt.show()
""")

# ── 5. GE with Brentq ──────────────────────────────────────────────────
md("""
## 3. GE Steady State: Brent vs Bisection

We solve for the equilibrium $r^*$ using two root-finding methods and compare convergence.
""")

code("""
# Brent's method
print("=" * 70)
print("Brent's method")
print("=" * 70)
result_brent = solve_aiyagari_ge(
    solver=EGM(tol=1e-8, max_iter=2000),
    income_process=income,
    a_grid=a_grid,
    beta=beta,
    gamma=gamma,
    alpha=alpha,
    delta=delta,
    r_bounds=r_bounds,
    method="brentq",
    verbose=True,
)
""")

# ── 6. GE with Bisection ───────────────────────────────────────────────
code("""
# Bisection method
print("=" * 70)
print("Bisection method")
print("=" * 70)
result_bisect = solve_aiyagari_ge(
    solver=EGM(tol=1e-8, max_iter=2000),
    income_process=income,
    a_grid=a_grid,
    beta=beta,
    gamma=gamma,
    alpha=alpha,
    delta=delta,
    r_bounds=r_bounds,
    method="bisection",
    verbose=True,
)
""")

# ── 7. Root-finder comparison ───────────────────────────────────────────
code("""
# Comparison table
print("\\nRoot-finder comparison")
print("-" * 70)
print(f"{'Method':<12s} {'r*':>12s} {'K*':>10s} {'Iters':>8s} {'Time(s)':>10s}")
print("-" * 70)
for name, res in [("Brent", result_brent), ("Bisection", result_bisect)]:
    print(f"{name:<12s} {res['r']:12.8f} {res['K']:10.4f} {res['n_iterations']:8d} {res['elapsed']:10.2f}")
print(f"\\n|r_brent - r_bisect| = {abs(result_brent['r'] - result_bisect['r']):.2e}")
""")

# ── 8. Multi-solver comparison header ──────────────────────────────────
md("""
## 4. Multi-Solver GE Comparison

Run the GE solver with each available solver backend and compare equilibrium values and timing.
""")

# ── 9. Run all solvers ──────────────────────────────────────────────────
code("""
solver_specs = [
    ("EGM", EGM(tol=1e-8, max_iter=2000)),
    ("GridVFI", GridVFI(tol=1e-8, max_iter=2000)),
    ("PFI", PolicyFunctionIteration(tol=1e-10, max_iter=500)),
    ("HowardGrid", HowardGrid(tol=1e-8, max_iter=200)),
    ("HowardImprovement", HowardImprovement(n_a_breaks=51, tol=1e-6, max_iter=120, n_warmup=2, verbose=False)),
    ("Collocation_Spline", CollocationVFI_Spline(n_a_breaks=51, tol=1e-8, max_iter=2000, verbose=False)),
]

ge_results = {}

for name, solver in solver_specs:
    print(f"\\nSolving GE with {name}...")
    res = solve_aiyagari_ge(
        solver=solver,
        income_process=income,
        a_grid=a_grid,
        beta=beta,
        gamma=gamma,
        alpha=alpha,
        delta=delta,
        r_bounds=r_bounds,
        method="brentq",
        verbose=True,
    )
    ge_results[name] = res
    print(f"  --> r*={res['r']:.8f}  K*={res['K']:.4f}  time={res['elapsed']:.2f}s")

print("\\nAll solvers complete.")
""")

# ── 10. Solver comparison table ─────────────────────────────────────────
code("""
print("\\nMulti-solver GE comparison")
print("=" * 110)
print(f"{'Solver':<22s} {'r*':>12s} {'K*':>10s} {'w*':>10s} {'Y*':>10s} {'C*':>10s} {'Y-dK-C':>12s} {'Iters':>6s} {'Time':>8s}")
print("=" * 110)
for name, res in ge_results.items():
    rc = res["Y"] - delta * res["K"] - res["aggregates"]["C"]
    print(
        f"{name:<22s} {res['r']:12.8f} {res['K']:10.4f} {res['w']:10.4f} "
        f"{res['Y']:10.4f} {res['aggregates']['C']:10.4f} {rc:12.4e} "
        f"{res['n_iterations']:6d} {res['elapsed']:8.2f}"
    )
""")

# ── 11. Diagnostics header ──────────────────────────────────────────────
md("""
## 5. Diagnostics

### 5.1 Policy Functions
""")

# ── 12. Policy functions ────────────────────────────────────────────────
code("""
# Use EGM equilibrium result for baseline plots
res_egm = ge_results["EGM"]
hh_eq = res_egm["household"]
sol_eq = res_egm["solved_policy"]
z_grid = income.z_grid

# Select low, mid, high income states
z_idx = [0, ne // 2, ne - 1]
z_labels = [f"z={z_grid[i]:.3f}" for i in z_idx]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for k, iz in enumerate(z_idx):
    ax1.plot(a_grid, sol_eq.policy_c[iz], label=z_labels[k], linewidth=1.5)
    ax2.plot(a_grid, sol_eq.policy_a[iz] - a_grid, label=z_labels[k], linewidth=1.5)

ax1.set_xlabel("Assets $k$")
ax1.set_ylabel("Consumption $c$")
ax1.set_title("Consumption Policy at Equilibrium")
ax1.legend()
ax1.grid(alpha=0.3)
ax1.set_xlim(0, 50)

ax2.set_xlabel("Assets $k$")
ax2.set_ylabel("Net savings $k' - k$")
ax2.set_title("Savings Policy at Equilibrium")
ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax2.legend()
ax2.grid(alpha=0.3)
ax2.set_xlim(0, 50)

plt.suptitle(f"EGM solution at r*={res_egm['r']:.6f}, w*={res_egm['w']:.4f}", fontsize=13)
plt.tight_layout()
plt.show()
""")

# ── 13. Wealth distribution ─────────────────────────────────────────────
md("""
### 5.2 Wealth Distribution
""")

code("""
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

da = np.diff(a_grid, prepend=a_grid[0] - (a_grid[1] - a_grid[0]))

for name, res in ge_results.items():
    g = res["distribution"]
    marg = g.sum(axis=0)
    # PDF (skip atom at k=0)
    density = marg[1:] / da[1:]
    ax1.plot(a_grid[1:], density, label=name, linewidth=1.5)
    # CDF
    cdf = np.cumsum(marg)
    cdf /= cdf[-1]
    ax2.plot(a_grid, cdf, label=name, linewidth=1.5)

# Annotate borrowing constraint mass
mass_at_zero = ge_results["EGM"]["distribution"].sum(axis=0)[0]
ax1.axvline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
ax1.set_xlim(0, 100)
ax1.set_xlabel("Wealth $k$")
ax1.set_ylabel("Density")
ax1.set_title(f"Wealth Distribution (PDF)\\n[mass at $k=0$: {mass_at_zero:.3f}]")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.25)

ax2.set_xlim(0, 200)
ax2.set_xlabel("Wealth $k$")
ax2.set_ylabel("CDF")
ax2.set_title("Wealth Distribution (CDF)")
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.show()
""")

# ── 14. Euler residuals ─────────────────────────────────────────────────
md("""
### 5.3 Euler Residuals
""")

code("""
print("Euler residuals (unconstrained states)")
print("-" * 70)
print(f"{'Solver':<22s} {'max':>12s} {'median':>12s} {'p95':>12s}")
print("-" * 70)

for name, res in ge_results.items():
    hh = res["household"]
    euler = hh.euler_check()
    unconstrained = res["solved_policy"].policy_a > (a_grid[0] + 1e-10)
    res_unc = np.abs(euler[unconstrained])
    if res_unc.size > 0:
        print(
            f"{name:<22s} {res_unc.max():12.4e} "
            f"{np.median(res_unc):12.4e} {np.quantile(res_unc, 0.95):12.4e}"
        )
    else:
        print(f"{name:<22s} no unconstrained states")
""")

# ── 15. Multi-solver policy overlay ─────────────────────────────────────
md("""
### 5.4 Policy Function Comparison Across Solvers
""")

code("""
# Overlay at median income state
iz_mid = ne // 2
a_zoom = 50  # zoom region

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

for name, res in ge_results.items():
    sol = res["solved_policy"]
    ax1.plot(a_grid, sol.policy_c[iz_mid], label=name, linewidth=1.5)
    ax2.plot(a_grid, sol.policy_a[iz_mid] - a_grid, label=name, linewidth=1.5)

ax1.set_xlim(0, a_zoom)
ax1.set_xlabel("Assets $k$")
ax1.set_ylabel("Consumption")
ax1.set_title(f"Consumption policy at z={z_grid[iz_mid]:.3f}")
ax1.legend(fontsize=8)
ax1.grid(alpha=0.25)

ax2.set_xlim(0, a_zoom)
ax2.set_xlabel("Assets $k$")
ax2.set_ylabel("Net savings")
ax2.set_title(f"Savings policy at z={z_grid[iz_mid]:.3f}")
ax2.axhline(0, color="black", linestyle="--", linewidth=0.5)
ax2.legend(fontsize=8)
ax2.grid(alpha=0.25)

plt.tight_layout()
plt.show()
""")

# ── Write notebook ──────────────────────────────────────────────────────
nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11.0"},
    },
    "cells": cells,
}

out = pathlib.Path(__file__).parent / "pset5_aiyagari_ge.ipynb"
out.write_text(json.dumps(nb, indent=1))
print(f"Wrote {out} ({len(cells)} cells)")
