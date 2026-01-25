import numpy as np
from numba import njit, uint


''' Grids and discretization '''


def markov_chain_ar1(rho, sig, n):
    # approximates by a markov chain the AR(1) process in logs using Rowenhorst's method
    # log(z_t) = rho * z_{t - 1} + sig_e * N(0, 1)
    # input sig = sig_e / sqrt(1 - rho ** 2) is the stationary stardard deviation
    p = q = 0.5 * (1 + rho)
    Pi = np.array([[p, 1 - p],
                   [1 - q, q]])

    for k in range(2, n):
        Pi_new = np.zeros((k + 1, k + 1))
        Pi_new[:-1, :-1] += p * Pi
        Pi_new[:-1, 1:] += (1 - p) * Pi
        Pi_new[1:, :-1] += (1 - q) * Pi
        Pi_new[1:, 1:] += q * Pi
        Pi = Pi_new / np.sum(Pi_new, axis=1)[:, np.newaxis]

    pi = np.linalg.matrix_power(Pi.T, int(np.log(1e-15) / np.log(rho)))[:, 0]
    z = np.exp(sig * np.sqrt(n - 1) * np.linspace(-1, 1, n))
    z /= np.sum(pi * z)

    return z, Pi, pi


def make_grid(vmin, vmax, n, smallest_step=0.01):
    curv = np.log((vmax - vmin) / smallest_step) / np.log(n - 1)
    curv = np.maximum(curv, 1)
    return vmin + (vmax - vmin) * (np.linspace(0, 1, n) ** curv)


''' Solvers for Bellman equations '''


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


@njit
def solve_euler_2a_exo(c_foc_a, c_foc_b, budget, a_grid, b_grid, ra, div_rate=0.0):
    ne, na, nb = c_foc_a.shape

    a_a = np.zeros((ne, na, nb))
    b_a = np.zeros((ne, na, nb))

    ia_a = np.zeros((ne, na, nb), dtype=uint)
    ib_a = np.zeros((ne, na, nb), dtype=uint)

    wa_a = np.zeros((ne, na, nb))
    wb_a = np.zeros((ne, na, nb))

    a_n = np.zeros((ne, na, nb))
    b_n = np.zeros((ne, na, nb))

    ia_n = np.zeros((ne, na, nb), dtype=uint)
    ib_n = np.zeros((ne, na, nb), dtype=uint)

    wa_n = np.zeros((ne, na, nb))
    wb_n = np.zeros((ne, na, nb))

    for ie in range(ne):
        for ia in range(na):

            i_start_a = 0
            i_start_b = 0

            for ib in range(nb):
                idx = ie, ia, ib  # just to save notation

                coh = budget[idx]  # cash on hand here

                a_unc, ia_unc, wa_unc, b_unc, ib_unc, wb_unc = solve_a_unconstrained_b(c_foc_a[ie, :, :], c_foc_b[ie, :, :], coh, a_grid, b_grid, i_start_a=i_start_a)
                i_start_a = ia_unc

                # check if any constraints for b are being violated
                if b_unc < b_grid[0]:
                    a_a[idx], ia_a[idx], wa_a[idx] = solve_a_constrained_b(0, c_foc_a[ie, :, :], coh, a_grid, b_grid, i_start_a=i_start_a)
                    b_a[idx] = b_grid[0]
                    ib_a[idx] = 0
                    wb_a[idx] = 1.0
                elif b_unc > b_grid[-1]:
                    a_a[idx], ia_a[idx], wa_a[idx] = solve_a_constrained_b(-1, c_foc_a[ie, :, :], coh, a_grid, b_grid, i_start_a=i_start_a)
                    b_a[idx] = b_grid[-1]
                    ib_a[idx] = nb - 2
                    wb_a[idx] = 0.0
                else:
                    a_a[idx] = a_unc
                    ia_a[idx] = ia_unc
                    wa_a[idx] = wa_unc

                    b_a[idx] = b_unc
                    ib_a[idx] = ib_unc
                    wb_a[idx] = wb_unc

                # the minimum operator below is assuming that agents can consume what exceeds amax
                # a_n[idx] = np.minimum((1 + ra) * a_grid[ia], a_grid[-1])
                a_n[idx] = (1 - div_rate) * (1 + ra) * a_grid[ia]
                # solve the non-adjuster's problem
                b_n[idx], ib_n[idx], wb_n[idx], ia_n[idx], wa_n[idx] = solve_b_given_a(a_n[idx], c_foc_b[ie, :, :], coh, a_grid, b_grid,
                                                                                       i_start_a=ia, i_start_b=i_start_b, impose_constraint=True)

                i_start_b = ib_n[idx]

    c_a = budget - a_a - b_a
    c_n = budget - a_n - b_n

    return a_a, b_a, c_a, ia_a, ib_a, wa_a, wb_a, a_n, b_n, c_n, ia_n, ib_n, wa_n, wb_n


@njit
def solve_b_given_a(a, c_foc_b, coh, a_grid, b_grid, i_start_a=0, i_start_b=0, impose_constraint=False):
    # find where "a" is in a_grid
    i_a, w_a = interp_grid(a, a_grid, i_start_a)

    # consumption implied by the FOC for b at point a
    c_foc = w_a * c_foc_b[i_a, :] + (1 - w_a) * c_foc_b[i_a + 1, :]

    # consumption as a function of b
    c = coh - a - b_grid

    i_b, w_b = interp_grid(0.0, c_foc - c, i_start_b)

    if impose_constraint:
        w_b = np.minimum(np.maximum(w_b, 0.0), 1.0)

    b = w_b * b_grid[i_b] + (1 - w_b) * b_grid[i_b + 1]

    return b, i_b, w_b, i_a, w_a


@njit
def solve_a_unconstrained_b(c_foc_a, c_foc_b, coh, a_grid, b_grid, i_start_a=0):
    na = len(a_grid)

    # optimal unconstrained b for a given a = a_grid[iap]
    b_unc_given_a = np.zeros(na)

    # c implied by the first order condition of a and actual c
    c_given_a = np.zeros(na)
    c_foc_given_a = np.zeros(na)

    i_start_b = 0

    for iap in range(i_start_a, na):
        # solve for unconstrained b given a = a_grid[iap]
        b_unc_given_a[iap], i_b, w_b, _, _ = solve_b_given_a(a_grid[iap], c_foc_b, coh, a_grid, b_grid, i_start_a=iap, i_start_b=i_start_b)

        # implied consumption
        c_given_a[iap] = coh - b_unc_given_a[iap] - a_grid[iap]
        c_foc_given_a[iap] = w_b * c_foc_a[iap, i_b] + (1 - w_b) * c_foc_a[iap, i_b + 1]

        if c_foc_given_a[iap] > c_given_a[iap]:
            break

        i_start_b = i_b

    if iap == 0:  # constrained at amin
        i_a = 0
        w_a = 1.0
    elif (iap == na - 1) and (c_foc_given_a[iap] <= c_given_a[iap]):  # constrained at amax
        i_a = na - 2
        w_a = 0.0
    else:
        y0 = c_given_a[iap - 1] - c_foc_given_a[iap - 1]
        y1 = c_given_a[iap] - c_foc_given_a[iap]
        i_a = iap - 1
        w_a = y1 / (y1 - y0)

    a = w_a * a_grid[i_a] + (1 - w_a) * a_grid[i_a + 1]
    b = w_a * b_unc_given_a[i_a] + (1 - w_a) * b_unc_given_a[i_a + 1]

    i_b, w_b = interp_grid(b, b_grid, i_b)

    return a, i_a, w_a, b, i_b, w_b


@njit
def solve_a_constrained_b(ibp, c_foc_a, coh, a_grid, b_grid, i_start_a=0):
    # consumption as a function of a
    c = coh - a_grid - b_grid[ibp]

    i_a, w_a = interp_grid(0.0, c_foc_a[:, ibp] - c, i_start_a)

    w_a = np.minimum(np.maximum(w_a, 0.0), 1.0)
    a = w_a * a_grid[i_a] + (1 - w_a) * a_grid[i_a + 1]

    return a, i_a, w_a


@njit
def update_Va_not_adjust(EVa, beta, ra, ia_n, wa_n, ib_n, wb_n):
    # updates the marginal utility of illiquid assets for non adjusters
    ne, na, nb = EVa.shape
    Va_not = np.zeros((ne, na, nb))

    for ie in range(ne):
        for ia in range(na):
            for ib in range(nb):
                iap = ia_n[ie, ia, ib]
                ibp = ib_n[ie, ia, ib]
                wa = wa_n[ie, ia, ib]
                wb = wb_n[ie, ia, ib]

                Va_not[ie, ia, ib] = wa * wb * EVa[ie, iap, ibp] + wa * (1 - wb) * EVa[ie, iap, ibp + 1] +\
                                     (1 - wa) * wb * EVa[ie, iap + 1, ibp] + (1 - wa) * (1 - wb) * EVa[ie, iap + 1, ibp + 1]

    Va_not = (1 + ra) * beta * Va_not

    return Va_not


''' Utilities for distributions '''


@njit
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


@njit
def update_exogenous_1a(Pi, g_prev):
    return Pi.T @ g_prev


@njit
def update_policy_shock_1a(a_ind, da_ind_w, g_prev):
    ne, na = a_ind.shape
    dg = np.zeros_like(g_prev)

    for ie in range(ne):
        for ia in range(na):
            dg[ie, a_ind[ie, ia]] += da_ind_w[ie, ia] * g_prev[ie, ia]
            dg[ie, a_ind[ie, ia] + 1] -= da_ind_w[ie, ia] * g_prev[ie, ia]

    return dg


@njit
def expectations_policy_1a(a_ind, a_ind_w, x):
    ne, na = a_ind.shape
    x_new = np.zeros((ne, na))

    for ie in range(ne):
        for ia in range(na):
            x_new[ie, ia] = a_ind_w[ie, ia] * x[ie, a_ind[ie, ia]] + (1 - a_ind_w[ie, ia]) * x[ie, a_ind[ie, ia] + 1]

    return x_new


@njit
def expectations_exogenous_1a(Pi, x):
    return Pi @ x


@njit
def update_policy_2a_exo(ia_a, ib_a, wa_a, wb_a, ia_n, ib_n, wa_n, wb_n, prob, g_prev):
    ne, na, nb = g_prev.shape
    g = np.zeros((ne, na, nb))

    for ie in range(ne):
        for ia in range(na):
            for ib in range(nb):
                idx = ie, ia, ib

                # update adjusters
                iap = ia_a[idx]
                ibp = ib_a[idx]
                wa = wa_a[idx]
                wb = wb_a[idx]

                g[ie, iap, ibp] += prob * wa * wb * g_prev[idx]
                g[ie, iap + 1, ibp] += prob * (1 - wa) * wb * g_prev[idx]
                g[ie, iap, ibp + 1] += prob * wa * (1 - wb) * g_prev[idx]
                g[ie, iap + 1, ibp + 1] += prob * (1 - wa) * (1 - wb) * g_prev[idx]

                # update non-adjusters
                iap = ia_n[idx]
                ibp = ib_n[idx]
                wa = wa_n[idx]
                wb = wb_n[idx]

                g[ie, iap, ibp] += (1 - prob) * wa * wb * g_prev[idx]
                g[ie, iap + 1, ibp] += (1 - prob) * (1 - wa) * wb * g_prev[idx]
                g[ie, iap, ibp + 1] += (1 - prob) * wa * (1 - wb) * g_prev[idx]
                g[ie, iap + 1, ibp + 1] += (1 - prob) * (1 - wa) * (1 - wb) * g_prev[idx]

    return g


def update_exogenous_2a_exo(Pi, g):
    return np.einsum('ij,i...->j...', Pi, g)


@njit
def expectations_policy_2a_exo(ia_a, ib_a, wa_a, wb_a, ia_n, ib_n, wa_n, wb_n, prob, x):
    ne, na, nb = x.shape
    x_new = np.zeros((ne, na, nb))

    for ie in range(ne):
        for ia in range(na):
            for ib in range(nb):
                # compute expectation if adjust
                iap = ia_a[ie, ia, ib]
                ibp = ib_a[ie, ia, ib]
                wa = wa_a[ie, ia, ib]
                wb = wb_a[ie, ia, ib]

                x_a = wa * wb * x[ie, iap, ibp] + (1 - wa) * wb * x[ie, iap + 1, ibp] +\
                      wa * (1 - wb) * x[ie, iap, ibp + 1] + (1 - wa) * (1 - wb) * x[ie, iap + 1, ibp + 1]

                # compute expectation if don't adjust
                iap = ia_n[ie, ia, ib]
                ibp = ib_n[ie, ia, ib]
                wa = wa_n[ie, ia, ib]
                wb = wb_n[ie, ia, ib]

                x_n = wa * wb * x[ie, iap, ibp] + (1 - wa) * wb * x[ie, iap + 1, ibp] +\
                      wa * (1 - wb) * x[ie, iap, ibp + 1] + (1 - wa) * (1 - wb) * x[ie, iap + 1, ibp + 1]

                x_new[ie, ia, ib] = prob * x_a + (1 - prob) * x_n

    return x_new


def expectations_exogenous_2a_exo(Pi, x):
    return np.einsum('ij,j...->i...', Pi, x)


@njit
def update_policy_shock_2a_exo(ia_a, ib_a, wa_a, wb_a, dwa_a, dwb_a,
                               ia_n, ib_n, wa_n, wb_n, dwa_n, dwb_n, prob, g_prev):
    ne, na, nb = g_prev.shape
    g = np.zeros((ne, na, nb))

    for ie in range(ne):
        for ia in range(na):
            for ib in range(nb):
                idx = ie, ia, ib

                # update adjusters
                iap = ia_a[idx]
                ibp = ib_a[idx]
                wa = wa_a[idx]
                wb = wb_a[idx]
                dwa = dwa_a[idx]
                dwb = dwb_a[idx]

                g[ie, iap, ibp] += prob * (dwa * wb + wa * dwb) * g_prev[idx]
                g[ie, iap + 1, ibp] += prob * (-dwa * wb + (1 - wa) * dwb) * g_prev[idx]
                g[ie, iap, ibp + 1] += prob * (dwa * (1 - wb) - wa * dwb) * g_prev[idx]
                g[ie, iap + 1, ibp + 1] += prob * (-dwa * (1 - wb) - (1 - wa) * dwb) * g_prev[idx]

                # update non-adjusters
                iap = ia_n[idx]
                ibp = ib_n[idx]
                wa = wa_n[idx]
                wb = wb_n[idx]
                dwa = dwa_n[idx]
                dwb = dwb_n[idx]

                g[ie, iap, ibp] += (1 - prob) * (dwa * wb + wa * dwb) * g_prev[idx]
                g[ie, iap + 1, ibp] += (1 - prob) * (-dwa * wb + (1 - wa) * dwb) * g_prev[idx]
                g[ie, iap, ibp + 1] += (1 - prob) * (dwa * (1 - wb) - wa * dwb) * g_prev[idx]
                g[ie, iap + 1, ibp + 1] += (1 - prob) * (-dwa * (1 - wb) - (1 - wa) * dwb) * g_prev[idx]

    return g


''' General numerical utilities '''


@njit
def interp_grid(v, x, i_start=0):
    # given an increasing grid x, find index ind such that x[ind] <= v < x[ind + 1]
    # and a weight such that v = weight * x[ind] + (1 - weight) * x[ind + 1]
    # if v < x[0] or v > x[-1], extrapolate linearly

    n = len(x)
    i = i_start
    # check if needs to go up or down the grid starting from i_start
    if v > x[i_start]:
        for i in range(i_start + 1, n):
            if x[i] > v:
                break
        ind = i - 1
    else:
        for i in range(i_start - 1, -1, -1):
            if x[i] <= v:
                break
        ind = i
    weight = (x[ind + 1] - v) / (x[ind + 1] - x[ind])

    return ind, weight

