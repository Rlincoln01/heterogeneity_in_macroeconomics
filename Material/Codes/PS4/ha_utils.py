

# Import Libraries
import numpy as np
import scipy.stats as stats
from scipy.stats import norm
from scipy.special import binom
from numba import njit, uint
from scipy.interpolate import interp1d
from scipy.optimize   import brentq

# 1. Grids and Discretization of Income Processes ----------------------------------------------------

def power_grid(amin,amax,n,phi=25,nu=5):
    """
    Builds a Power (nonlinear) grid for a given variable of interest (in this case, assets)

    Parameters:
    # amin - minimum value of the grid
    # amax - maximum value of the grid
    # n - Size of the Grid
    # phi - controls the density in the lower part of the grid
    # nu - controls the density in the upper part of the grid
    """

    # 1st step: Construct an n-point equidistant grid
    x_grid = np.linspace(0, 1,n)

    # 2nd step: Apply the non-linear transformation of the grid
    xx_grid = x_grid + phi*x_grid**nu
    xmax = max(xx_grid)
    xmin = min(xx_grid)

    # 3rd step: Obtain the final grid
    a_grid = amin + ((amax-amin)/(xmax-xmin))*xx_grid

    return a_grid
    
def rouwenhorst_AR1(mu,rho, sigma_e, N):
    """
    Builds the transition matrix and the income grid for an AR(1) process using the Rouwenhorst method.

    Parameters:
    # mu - mean of the AR(1) process
    # rho - persitence of AR(1)
    # sigma_e - standard deviation of the epsilon on AR(1)
    # N - Size of the grid
    """

    #  === Define the parameters which build the discretization === #

    # Grid
    phi = np.sqrt(N-1)*(sigma_e/(np.sqrt(1-rho**2)))
    z_grid = np.linspace(-phi, phi, N) + mu

    # Values of p and q for AR(1) process
    p = (1+ rho)/2
    q = p

    # Initialize the coefficient matrix Pi_N with zeros
    Pi_N = np.zeros((N, N))

    # Iterate over i = 1, 2, ..., N
    for i in range(1, N+1):
        # Initialize a temporary array to store the combined coefficients
        temp_coeffs = np.zeros(N)

        # First expansion: [p + (1 - p)t]^{N-i}
        for k in range(N-i+1):    # (N-i) + (1) - because of range
            coeff1 = binom(N-i, k) * (p**(N-i-k)) * ((1-p)**k)
            # Second expansion: (1 - q + qt)^{i-1}
            for m in range(i):    # (i-1) + (1) - because of range
                coeff2 = binom(i-1, m) * ((1-q)**(i-1-m)) * (q**m)
                # Combine terms for t^(k+m)
                if (k + m) < N:
                    temp_coeffs[k + m] += coeff1 * coeff2

        # Store the coefficients for Φ(t; N, i) in the matrix
        Pi_N[i-1, :] = temp_coeffs

    # Sanity check: Ensure row sums are 1
    row_sums = np.sum(Pi_N, axis=1)
    if not np.allclose(row_sums, 1):
      raise ValueError("Error: Row sums in the transition matrix are not equal to 1.")


    return Pi_N, z_grid    


def tauchen_AR1(mu,rho,sigma_e,N,m=3):
    """
    Constructs the N×N transition matrix and the N-point grid for the
    AR(1) process using Tauchen’s (1986) method.

    Parameters
    ----------
    mu       : float   – unconditional mean of y_t
    rho      : float   – AR(1) persistence coefficient  (|ρ| < 1)
    sigma_e  : float   – std. dev. of the i.i.d. shock ε_t
    N        : int     – number of grid points
    m        : float   – width parameter (default 3 → ±3 σ_y)

    Returns
    -------
    Pi       : (N,N)  Numpy array – transition‐probability matrix
    y_grid   : (N,)   Numpy array – state grid for y
    """

    # 1.  Grid construction ------------------------------------------------
    sigma_y = sigma_e / np.sqrt(1.0 - rho**2)      # unconditional σ_y
    y_max   =  m * sigma_y
    y_min   = -m * sigma_y
    d       = (y_max - y_min) / (N - 1)            # grid spacing

    y_grid  = np.linspace(y_min, y_max, N) + mu    # shift by μ

    # 2.  Transition matrix -----------------------------------------------
    Pi = np.zeros((N, N))

    for j in range(N):               # current state index
        for k in range(N):           # next-state index

            if k == 0:               # lower boundary
                z_upper = (y_grid[0] + d/2.0 - rho * (y_grid[j])) / sigma_e
                Pi[j, k] = norm.cdf(z_upper)

            elif k == N-1:           # upper boundary
                z_lower = (y_grid[-1] - d/2.0 - rho * (y_grid[j])) / sigma_e
                Pi[j, k] = 1.0 - norm.cdf(z_lower)

            else:                    # interior points
                z_upper = (y_grid[k]   + d/2.0 - rho * (y_grid[j])) / sigma_e
                z_lower = (y_grid[k]   - d/2.0 - rho * (y_grid[j])) / sigma_e
                Pi[j, k] = norm.cdf(z_upper) - norm.cdf(z_lower)

    # 3.  (Optional) sanity check – rows must sum to 1
    if not np.allclose(Pi.sum(axis=1), 1.0):
        raise RuntimeError("Tauchen: rows of transition matrix do not sum to 1.")

    return Pi, y_grid


# 2. Solvers for Bellman Equations --------------------------------------------------

@njit
def solve_euler_one_asset(c_foc, budget, a_grid):
    ne, na = c_foc.shape

    a = np.zeros((ne, na))  # next period assets
    a_ind = np.zeros((ne, na), dtype=uint)  # index of next period assets in asset grid
    # a_ind = np.zeros((ne, na), dtype=int)
    a_ind_w = np.zeros((ne, na))  # weight of a_ind in the linear interpolation

    # loop over grid to find optimal consumption
    for ie in range(ne):
        # initially, start searching at the first grid point
        ia_start = 0
        for ia in range(na):
            # loop over asset grid point to find optimal policy
            for iap in range(ia_start, na):
                c_here = budget[ie, ia] - a_grid[iap]  # if go to point iap of a_grid, this is the implied consumption

                if c_foc[ie, iap] > c_here:
                    break

            if iap == 0:  # constrained at amin?
                a_ind[ie, ia] = 0
                a_ind_w[ie, ia] = 1.0
            elif (iap == na - 1) and (c_foc[ie, iap] <= c_here):  # constrained at amax?
                a_ind[ie, ia] = na - 2
                a_ind_w[ie, ia] = 0.0
            else:
                # otherwise, interpolate to find optimal a'
                c_prev = budget[ie, ia] - a_grid[iap - 1]
                y0 = c_prev - c_foc[ie, iap - 1]
                y1 = c_here - c_foc[ie, iap]
                a_ind[ie, ia] = iap - 1
                a_ind_w[ie, ia] = y1 / (y1 - y0)

            a[ie, ia] = a_ind_w[ie, ia] * a_grid[a_ind[ie, ia]] + (1 - a_ind_w[ie, ia]) * a_grid[a_ind[ie, ia] + 1]
            ia_start = iap

    c = budget - a

    return c, a, a_ind, a_ind_w


# 3. Utilities for Distributions ----------------------------------------------------

def compute_joint_transition_matrix(a_ind, a_ind_w, Pi):
    """
    Computes the joint transition matrix for the income-wealth distribution.
    
    Parameters:
    - a_ind: (nz, na) array with indices for next-period asset grid
    - a_ind_w: (nz, na) array with interpolation weights for a_ind (mass)
    - Pi: (nz, nz) income transition matrix

    Returns:
    - T: (nz*na, nz*na) joint transition matrix for the income-wealth distribution.
    """
    nz, na = a_ind.shape
    T = np.zeros((nz * na, nz * na))
    
    for iz in range(nz):
        for ia in range(na):
            current_index = iz * na + ia
            
            for iz_next in range(nz):
                prob_income = Pi[iz, iz_next]

                low_index = int(a_ind[iz, ia])
                high_index = low_index + 1

                # Check for boundary conditions: if at the last asset grid point,
                # mass goes entirely to that state.
                if high_index >= na:
                    T[current_index, iz_next * na + (na - 1)] += prob_income
                else:
                    T[current_index, iz_next * na + low_index] += prob_income * a_ind_w[iz, ia]
                    T[current_index, iz_next * na + high_index] += prob_income * (1 - a_ind_w[iz, ia])
    return T


def update_policy_1a(a_ind, a_ind_w, g_prev):
    # this function updates the distribution of agents given the optimal policy
    # then it the mass of the point (ie, ia) to (ie, a[ie, ia])

    ne, na = a_ind.shape
    g = np.zeros((ne, na))

    for ie in range(ne):
        for ia in range(na):
            g[ie, a_ind[ie, ia]] += a_ind_w[ie, ia] * g_prev[ie, ia]
            g[ie, a_ind[ie, ia] + 1] += (1 - a_ind_w[ie, ia]) * g_prev[ie, ia]

    return g

def stationary_dist(P):
    """
    Stationary distribution π solving  π' P = π',  π' 1 = 1,
    obtained from the null-space of (I-P)ᵀ with an additional
    normalising equation.
    Works robustly for large N.

    Parameters
    ----------
    P : (N,N) ndarray – row-stochastic transition matrix.

    Returns
    -------
    π : (N,) ndarray – stationary distribution (non-negative, sums to one).
    """

    P = np.asarray(P)
    if P.ndim != 2 or P.shape[0] != P.shape[1]:
        raise ValueError("P must be a square matrix.")
    if not np.allclose(P.sum(axis=1), 1.0):
        raise ValueError("Rows of P must sum to 1.")

    N = P.shape[0]

    # Build augmented system:              (I - Pᵀ) π = 0
    # add normalising condition            1' π  = 1
    A = np.vstack((np.eye(N) - P.T, np.ones((1, N))))
    b = np.zeros(N + 1)
    b[-1] = 1.0

    # Solve the (N+1)×N least-squares system
    π, *_ = np.linalg.lstsq(A, b, rcond=None)

    # Small negative numerical noise → clip
    π = np.maximum(π, 0)
    π = π / π.sum()          # ensure Σπ = 1

    return π


# An alternative stationary distribution function using eigen decomposition
def stationary_dist_eig(P):
    # Compute eigenvalues & eigenvectors of the transpose
    w, v = np.linalg.eig(P.T)
    # Find the eigenvector associated with eigenvalue 1
    idx = np.argmin(np.abs(w - 1))
    pi = np.real(v[:, idx])
    # Ensure non-negative and normalize
    pi = pi.clip(min=0)
    return pi / pi.sum()


# 3. Simulation Functions --------------------------------------------------------------------------------


def simulate_MC(P, z_grid, T, start_state=None, **kwargs):
    """
    Simulates a path from a Markov chain given the transition matrix P and state space z_grid.

    Parameters:
    P          : numpy array of shape (N, N) - The transition matrix
    Y_N        : numpy array of shape (N,)   - The state space grid
    T          : int                         - Number of periods to simulate
    start_state: int or None                 - starting state index, default is None (random start)

    Returns:
    path       : numpy array of shape (T,)   - The simulated path

    Optional (kwargs):
    burn-in    : int                         - Percentage of initial periods to discard of the simulation
    seed       : int                         - Seed for the random number generator
    """

    seed = kwargs.get('seed', None)  # Get seed from kwargs, default to None
    burn_in_percentage = kwargs.get('burn_in', 0)  # Default to 0% if not specified

    # make one local RNG
    if seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(seed)

    N = P.shape[0]  # Number of states

    # If no start state is given, randomly choose one
    if start_state is None:
        # current_state = np.random.choice(np.arange(N))
        current_state = rng.choice(np.arange(N))
    else:
        current_state = start_state

    # Initialize the array to store the path
    path = np.zeros(T)

    # Simulate the path
    for t in range(T):
        path[t] = z_grid[current_state]  # Record the current state in the path
        # Draw the next state based on the transition probabilities
        current_state = rng.choice(np.arange(N), p=P[current_state])


    # Optional Burn-in:

    burn_in_period = int(burn_in_percentage / 100 * T)
    path = path[burn_in_period:]


    return path