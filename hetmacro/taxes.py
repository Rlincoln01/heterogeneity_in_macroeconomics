"""HSV (Heathcote-Storesletten-Violante) progressive tax functions.

The HSV tax schedule is T(y) = y - (1 - tau)/(1 - xi) * y^{1 - xi} - iota,
where tau governs the average level of marginal tax rates, xi governs the
slope (progressivity), and iota is a lump-sum transfer that balances the
government budget.

Special cases:
    xi = 0 : flat-rate tax  T(y) = tau * y - iota
    xi > 0 : progressive    T'(y) increasing in y

References
----------
Heathcote, Storesletten, Violante (2017). "Optimal Tax Progressivity."
Boar, Midrigan (2022). "Efficient Redistribution." JME.
"""

import numpy as np


def hsv_post_tax(y, tau, xi, iota):
    """Post-tax income: yhat = (1 - tau)/(1 - xi) * y^{1 - xi} + iota.

    Parameters
    ----------
    y : array_like
        Pre-tax income (must be > 0 for xi > 0).
    tau : float
        Average marginal tax rate parameter.
    xi : float
        Progressivity parameter (0 = flat, > 0 = progressive).
    iota : float
        Lump-sum transfer.

    Returns
    -------
    yhat : ndarray
        Post-tax income, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    y_safe = np.maximum(y, 1e-15)

    if abs(xi) < 1e-12:
        # Flat-rate limit: yhat = (1 - tau) * y + iota
        return (1.0 - tau) * y + iota

    return (1.0 - tau) / (1.0 - xi) * y_safe ** (1.0 - xi) + iota


def hsv_marginal_tax(y, tau, xi):
    """Marginal tax rate: T'(y) = 1 - (1 - tau) * y^{-xi}.

    Parameters
    ----------
    y : array_like
        Pre-tax income (must be > 0 for xi > 0).
    tau : float
        Average marginal tax rate parameter.
    xi : float
        Progressivity parameter.

    Returns
    -------
    mtax : ndarray
        Marginal tax rate, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    y_safe = np.maximum(y, 1e-15)

    if abs(xi) < 1e-12:
        return np.full_like(y, tau)

    return 1.0 - (1.0 - tau) * y_safe ** (-xi)


def hsv_post_tax_and_marginal(y, tau, xi, iota):
    """Compute both post-tax income and marginal tax rate efficiently.

    Shares the y^{-xi} computation between the two quantities.

    Parameters
    ----------
    y : array_like
        Pre-tax income.
    tau, xi, iota : float
        Tax parameters.

    Returns
    -------
    yhat : ndarray
        Post-tax income.
    mtax : ndarray
        Marginal tax rate.
    """
    y = np.asarray(y, dtype=float)
    y_safe = np.maximum(y, 1e-15)

    if abs(xi) < 1e-12:
        yhat = (1.0 - tau) * y + iota
        mtax = np.full_like(y, tau)
        return yhat, mtax

    temp = y_safe ** (-xi)  # expensive, compute once
    mtax = 1.0 - (1.0 - tau) * temp
    yhat = (1.0 - tau) / (1.0 - xi) * y_safe * temp + iota
    # Note: y^{1-xi} = y * y^{-xi} = y * temp
    return yhat, mtax


def hsv_tax(y, tau, xi, iota):
    """Tax paid: T(y) = y - (1 - tau)/(1 - xi) * y^{1 - xi} - iota.

    Parameters
    ----------
    y : array_like
        Pre-tax income.
    tau, xi, iota : float
        Tax parameters.

    Returns
    -------
    tax : ndarray
        Tax paid, same shape as y.
    """
    return np.asarray(y, dtype=float) - hsv_post_tax(y, tau, xi, iota)


def hsv_average_tax(y, tau, xi, iota):
    """Average tax rate: T(y) / y.

    Parameters
    ----------
    y : array_like
        Pre-tax income.
    tau, xi, iota : float
        Tax parameters.

    Returns
    -------
    avg_tax : ndarray
        Average tax rate, same shape as y.
    """
    y = np.asarray(y, dtype=float)
    y_safe = np.maximum(y, 1e-15)
    return hsv_tax(y, tau, xi, iota) / y_safe


def hsv_iota_balanced_budget(y_all, D, tau, xi, G):
    """Compute lump-sum transfer iota from the government balanced budget.

    The government budget constraint is:
        integral T(y_i) dF = G
        integral [y_i - (1-tau)/(1-xi) * y_i^{1-xi}] dF - iota = G
        iota = integral [y_i - (1-tau)/(1-xi) * y_i^{1-xi}] dF - G

    Parameters
    ----------
    y_all : ndarray
        Pre-tax income for every state (e, a), shape (n_e, n_a).
    D : ndarray
        Joint distribution over (e, a), shape (n_e, n_a), sums to 1.
    tau : float
        Average marginal tax rate parameter.
    xi : float
        Progressivity parameter.
    G : float
        Exogenous government spending.

    Returns
    -------
    iota : float
        Lump-sum transfer that balances the budget.
    """
    y_all = np.asarray(y_all, dtype=float)
    D = np.asarray(D, dtype=float)
    y_safe = np.maximum(y_all, 1e-15)

    if abs(xi) < 1e-12:
        # Flat rate: T(y) = tau * y - iota
        # integral(tau * y)dF - iota = G => iota = tau * E[y] - G
        tax_revenue_before_iota = float(np.sum(D * tau * y_all))
    else:
        # T(y) = y - (1-tau)/(1-xi) * y^{1-xi} - iota
        # integral[y - (1-tau)/(1-xi)*y^{1-xi}]dF - iota = G
        tax_revenue_before_iota = float(
            np.sum(D * (y_all - (1.0 - tau) / (1.0 - xi) * y_safe ** (1.0 - xi)))
        )

    return tax_revenue_before_iota - G
