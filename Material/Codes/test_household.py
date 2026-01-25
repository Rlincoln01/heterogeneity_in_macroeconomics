# %% import
import numpy as np
import ha_utils as ha_utils
import matplotlib.pyplot as plt
from olg_model import HouseholdOLG


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
beta = 0.95  # discount factor
crra = 2.0  # crra

# age-dependent objects
T = 80  # periods lived
T_ret = 45  # retirement age

p_death = np.zeros(T)  # probability of death
p_death[T_ret:] = 0.4 * (np.arange(T - T_ret) / (T - T_ret)) ** 4
p_death[-1] = 1.0
assert np.isclose(p_death[-1], 1.0)

z_age = np.exp(0.01 * np.arange(T_ret - 1))  # age component of labor skill

b_ret = 0.7  # social security benefits received after retirement

# prices etc.
w = 1.0  # real wage
tau = 0.0  # income tax
r = 0.04  # interest rate
gc = 0.01  # growth rate of new cohort size (or population in steady state)
transfer = 0


# %%  test steady state
            
model = HouseholdOLG(a_grid, z_grid, Pi, beta, crra, w, tau, r, b_ret, z_age, p_death, gc, transfer, T, T_ret)
model.steady_state()


# %%  steady state plots

a = model.outputs['a']
c = model.outputs['c']
e = model.outputs['e']
g = model.g

age_plot = [0, 20, 40, 60]
imax = 30

fig, ax = plt.subplots()
for t in age_plot:
    ax.plot(a_grid[:imax], c[nz // 2, :imax, t])
ax.legend([f'age = {20 + t:.0f}' for t in age_plot])
ax.grid(alpha=0.5)
ax.set_xlabel('Assets')
ax.set_ylabel('Consumption')
plt.show()

a_by_age = np.sum(a * g, axis=(0, 1)) / np.sum(g, axis=(0, 1))
c_by_age = np.sum(c * g, axis=(0, 1)) / np.sum(g, axis=(0, 1))
e_by_age = np.sum(e * g, axis=(0, 1)) / np.sum(g, axis=(0, 1))

cohort_size = np.sum(g, axis=(0, 1))
cohort_size /= cohort_size[0]

age = 20 + np.arange(T)

fig, ax = plt.subplots()
ax.plot(age, e_by_age)
ax.grid(alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Income')
plt.show()

fig, ax = plt.subplots()
ax.plot(age, a_by_age)
ax.grid(alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Assets')
plt.show()

fig, ax = plt.subplots()
ax.plot(age, c_by_age)
ax.grid(alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Consumption')
plt.show()

fig, ax = plt.subplots()
ax.plot(age, cohort_size)
ax.grid(alpha=0.5)
ax.set_xlabel('Age')
ax.set_ylabel('Cohort size')
plt.show()

# %% test transition

T = 100
t = np.arange(T)

r_path = r - 0.01 * (1 - np.exp(-0.2 * t))
gc_path = gc - 0.002 * (1 - np.exp(-0.2 * t))

paths = {'r': r_path, 'gc': gc_path}

aggregate_paths = model.mit_shock(paths)

# %% transition plots

fig, ax = plt.subplots()
ax.plot(t, aggregate_paths['A'])
ax.grid(alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Assets per capita')
plt.show()

fig, ax = plt.subplots()
ax.plot(t, 100 * aggregate_paths['gp'])
ax.grid(alpha=0.5)
ax.set_xlabel('Time')
ax.set_ylabel('Population growth rate')
plt.show()


# %%
