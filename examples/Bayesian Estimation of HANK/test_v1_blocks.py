import numpy as np
import sequence_jacobian as sj
from sequence_jacobian import simple, het, solved
import sequence_jacobian.utilities as sj_utils

@simple
def construct_grids(amax, amin, rho_e, sd_e, nA, nS):
    a_grid = sj_utils.discretize.agrid(amax=amax, n=nA, amin=amin)
    e_grid, e_pdf, Pi = sj_utils.discretize.markov_rouwenhorst(rho=rho_e, sigma=sd_e, N=nS)
    return a_grid, e_grid, e_pdf, Pi

def income_incidence(e_grid, Y, T):
    y_grid = (Y - T) * e_grid
    return y_grid

def household_init(rpost, Y, T, sigma, a_grid, e_grid):
    coh = (1 + rpost) * a_grid[np.newaxis, :] + (Y - T) * e_grid[:, np.newaxis]
    Va = (1 + rpost) * (0.2 * coh) ** (-sigma)
    return Va

@het(exogenous='Pi', policy='a', backward='Va', backward_init=household_init,
     hetinputs=[income_incidence])
def household(Va_p, a_grid, acons, e_grid, y_grid, rpost, beta, sigma):
    """Single backward EGM step. Note: Va_p already includes Pi@Va expectation."""
    uc_nextgrid = beta * Va_p  # V1.0: het block applies Pi @ Va internally
    c_nextgrid = uc_nextgrid ** (-1 / sigma)
    lhs = c_nextgrid + a_grid[np.newaxis, :] - y_grid[:, np.newaxis]
    rhs = (1 + rpost) * a_grid
    a = sj.interpolate_y(lhs, rhs, a_grid)
    sj.setmin(a, acons)
    c = rhs[np.newaxis, :] + y_grid[:, np.newaxis] - a
    uc = c ** (-sigma)
    Va = (1 + rpost) * uc
    uce = uc * e_grid[:, np.newaxis]
    return Va, a, c, uc, uce

@solved(unknowns={'C': 1., 'A': 1.}, targets=['Cres', 'Ares'], solver="broyden_custom")
def household_ra(C, A, r, rpost, Y, T, beta, sigma):
    Cres = beta * (1 + r) * C(1)**(-sigma) - C**(-sigma)
    Ares = (1 + rpost) * A(-1) + Y - T - C - A
    return Cres, Ares

@simple
def rpost_simple(r):
    rpost = r(-1)
    return rpost

@simple
def monetary_taylor(pi, ishock, rss, phi_pi):
    i = rss + phi_pi * pi + ishock
    r = i - pi(1)
    return i, r

@simple
def nkpc(pi, Y, X, C, kappa_w, vphi, frisch, markup_ss, eis, beta):
    piw = pi + X - X(-1)
    N = Y / X
    piwres = kappa_w * (vphi * N**(1/frisch) - 1/markup_ss * X * C**(-1/eis)) + beta * piw(1) - piw
    return piwres, piw, N

@simple
def mkt_clearing(A, B, Y, C, G):
    asset_mkt = A - B
    goods_mkt = C + G - Y
    return asset_mkt, goods_mkt

@solved(unknowns={'B': 0}, targets=['Bres'], solver="brentq")
def fiscal_TBrule(r, G, B, Tss, phi_T):
    T = Tss + phi_T * (B(-1) - B.ss)
    Bres = (1 + r(-1)) * B(-1) + G - T - B
    return T, Bres
