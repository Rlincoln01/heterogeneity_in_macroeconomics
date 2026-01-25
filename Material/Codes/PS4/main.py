"""
%--------------------------------------------------------------------------
% Macroeconomic Theory 2 - New York University
% Prof. Corina Boar - Spring 2025
% 
% Aiyagari Model - Discrete Time
% By: Rafael Lincoln
%
% Description: Code for the solution of the Aiyagari model using the EGM method
%              together with a newton method for the stationary equilibrium.
%              
% Date: 25 apr 2025
%--------------------------------------------------------------------------
"""


# Import Libraries
import numpy as np
import ha_utils as ha_utils
import matplotlib.pyplot as plt
import aiyagari as ay


# 1. Calibration/Parametrization ------------------------------------------

# Income Process
mu = 0.0  # mean of (log) income shocks
rho = 0.9  # persistence of (log) income shocks
sigma_e = 0.1  # standard deviation of (log)income shocks
nz = 7  # number of income states
Pi, z_grid = ha_utils.rouwenhorst_AR1(mu,rho, sigma_e, nz)


# Asset Grid
bl = 0.0  # lower bound of asset grid
ba = 20.0  # upper bound of asset grid
na = 300  # number of asset grid points
a_grid = ha_utils.power_grid(bl, ba, na)


# Model Parameters
beta = 0.96  # discount factor
gamma = 2.0  # coefficient of relative risk aversion
delta = 0.08  # depreciation rate
alpha = 0.36  # capital share in income


# 2. Solve the Household Problem ---------------------------------------------

# initial test for the household class in partial equilibrium:
r = 0.02
w = 1

hh = ay.Households(a_grid, z_grid, Pi, beta, r, w, gamma)

hh.steady_state()

# calculate aggregates
aggregates = hh.aggregates
# stationary distribution
stationary_dist = hh.g
# Policy_functions
pol_functions = hh.pol_functions

marginal_wealth = stationary_dist.sum(axis=0)


# Plot the marginal distribution of wealth for the lowest and highest income levels
fig, ax = plt.subplots(figsize=(8,6))

ax.plot(a_grid, marginal_wealth)
# ax.plot(a_grid, marginal_high, label="Highest Income Level")
ax.set_xlabel("Wealth")
ax.set_ylabel("Probability")
ax.set_ylim(0, 0.02)
ax.set_title("Marginal Wealth Distribution")
ax.legend()
ax.grid(True)
plt.show()

# Plot policy functions 

# Transpose consumption and savings
c_policy_egm = pol_functions['c'].T
sav_policy_egm = pol_functions['sav'].T
# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Consumption subplot
for z_idx in range(len(z_grid)):
    y_value = np.exp(z_grid[z_idx])  # Calculate y_t value
    ax1.plot(a_grid, c_policy_egm[:, z_idx], label=f'$y_t = {y_value:.2f}$')  # Use LaTeX formatting
ax1.set_xlabel('Wealth (a)')
# ax1.set_ylabel('Consumption (c)')
ax1.set_title('Consumption Policy Function')
ax1.legend()
ax1.grid(True)

# Savings subplot
for z_idx in range(len(z_grid)):
    y_value = np.exp(z_grid[z_idx])  # Calculate y_t value
    ax2.plot(a_grid, sav_policy_egm[:, z_idx], label=f'$y_t = {y_value:.2f}$')  # Use LaTeX formatting
ax2.set_xlabel('Wealth (a)')
# ax2.set_ylabel('Savings (s)')
ax2.set_title('Savings Policy Function')
ax2.legend()
ax2.axhline(0, color='black', linestyle='--')  # Add horizontal line at y=0
ax2.grid(True)

plt.tight_layout()
plt.show()

