# %% import
import numpy as np
import ha_utils as ha_utils
import matplotlib.pyplot as plt
from olg_model import solve_steady_state, HouseholdOLG


# %% define parameters

# asset grid
amin = 0
amax = 100
na = 500
a_grid = ha_utils.make_grid(amin, amax, na)

# labor skill grid
rho = 0.98  # persistence
sig = 0.4  # standard deviation of stationary distribution (not of shocks)
nz = 11
z_grid, Pi, _ = ha_utils.markov_chain_ar1(rho, sig, nz)  # grid and transition matrix

# preferences
crra = 2.0  # crra

# age-dependent objects
T = 80  # periods lived
T_ret = 45  # retirement age

p_death = np.zeros(T)  # probability of death
p_death[T_ret:] = 0.4 * (np.arange(T - T_ret) / (T - T_ret)) ** 4
p_death[-1] = 1.0
assert np.isclose(p_death[-1], 1.0)

z_age = np.exp(0.01 * np.arange(T_ret - 1))  # age component of labor skill

b_ret = 0.5  # social security benefits received after retirement

# prices etc.
r = 0.04  # interest rate
gc = 0.01  # growth rate of new cohort size (or population in steady state)

# government spending and debt as a fraction of GDP
G_Y = 0.15
B_Y = 0.6

# GDP, capital share, depreciation
Y = 1.0
alpha = 0.36
delta = 0.07


# %% solve steady state

diff, hh, out = solve_steady_state(a_grid, z_grid, Pi, crra, r, b_ret, z_age, p_death, gc, T, T_ret, alpha, delta, Y, B_Y, G_Y)


for k in out:
    print(f'{k} = {out[k]}')

# %% Calculate the demographic pyramid of the population

age_pyramid = [100*np.sum(hh.g[...,j]) for j in range(0,T)]

plt.bar(range(1,T+1), age_pyramid)
    
    



