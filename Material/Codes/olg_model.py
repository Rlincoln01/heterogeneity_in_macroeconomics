import numpy as np
import ha_utils
import scipy.optimize as opt


class HouseholdOLG():
    def __init__(self, a_grid, z_grid, Pi, beta, crra, w, tau, r, b_ret, z_age, p_death, gc, transfer, T, T_ret):
        self.a_grid = a_grid  # illiquid asset grid
        self.z_grid = z_grid  # labor productivity grid
        self.Pi = Pi          # transition matrix for z

        self.T = T            # number of periods lived
        self.T_ret = T_ret    # retirement age

        # put into a dict parameters that can be shocked
        self.inputs = {
            'beta': beta,          # discount factor
            'crra': crra,          # crra
            'w': w,                # real wage
            'tau': tau,            # taxes on earnings
            'r': r,                # interest rate
            'b_ret': b_ret,        # social security benefits parameters
            'gc': gc,              # population growth rate  
            'z_age': z_age,        # age-specific component of labor productivity
            'p_death': p_death,    # probability of dying
            'transfer': transfer,  # lump sum to households
        }

        self.outputs = None
        self.pol_interp = None
        self.EVa = None
        self.g = None
        self.aggregates = None

    def bellman_iteration(self, EVa_next=None, **kwargs):
        beta = kwargs['beta']
        crra = kwargs['crra']
        w = kwargs['w']
        tau = kwargs['tau']
        r = kwargs['r']
        b_ret = kwargs['b_ret']
        z_age = kwargs['z_age']
        p_death = kwargs['p_death']
        transfer = kwargs['transfer']

        z_grid = self.z_grid
        a_grid = self.a_grid
        Pi = self.Pi
        T = self.T
        T_ret = self.T_ret

        nz = len(self.z_grid)
        na = len(self.a_grid)

        EVa = np.zeros((nz, na, T))  # expected marginal utility of assets

        c = np.zeros((nz, na, T))  # average consumption
        a = np.zeros((nz, na, T))  # average illiquid asset demand
        e = np.zeros((nz, na, T))  # earnings
        l = np.zeros((nz, na, T))  # labor supply
        rev = np.zeros((nz, na, T))  # government net revenue
        beq = np.zeros((nz, na, T))  # unintended bequests

        # interpolation stuff for updating the distribution
        a_i = np.zeros((nz, na, T), dtype=int)
        a_w = np.zeros((nz, na, T))

        # loop over age
        for t in range(T - 1, -1, -1):
            # print(t)

            if t < T_ret - 1: # if not retired
                # earnings grid
                e_grid = (1 - tau) * w * z_grid * z_age[t]
                rev[..., t] = tau * w * z_age[t] * z_grid[:, np.newaxis]
                l[..., t] = z_age[t] * z_grid[:, np.newaxis]
            else:
                e_grid = (1 - tau) * b_ret * w * z_grid * z_age[-1]
                rev[..., t] = -(1 - tau) * b_ret * w * z_age[-1] * z_grid[:, np.newaxis]

            # household wealth
            budget = e_grid[:, np.newaxis] + (1 + r) * (a_grid[np.newaxis, :] + transfer)

            if t == (T - 1):
                # if at the last period, the marginal value of carrying assets is zero
                EVa_here = np.zeros((nz, na))
            else:
                if EVa_next is None:
                    EVa_here = EVa[..., t + 1]
                else:
                    EVa_here = EVa_next[..., t + 1]

            # consumption implied by the FOCs
            c_foc = np.maximum((1 - p_death[t]) * beta * EVa_here, 1e-12) ** (-1 / crra)

            c[..., t], a[..., t], a_i[..., t], a_w[..., t] = ha_utils.solve_euler_one_asset(c_foc, budget, a_grid)
            e[..., t] = e_grid[:, np.newaxis]
            beq[..., t] = p_death[t] * a[..., t]

            Va = (1 + r) * c[..., t] ** (-crra)

            if t < T_ret - 1:
                Pi_here = Pi
            else:
                Pi_here = np.eye(nz)

            EVa[..., t] = Pi_here @ Va

        outputs = {
            'a': a,
            'a_m': a_grid[np.newaxis, :, np.newaxis],
            'c': c,
            'e': e,
            'l': l,
            'rev': rev,
            'beq': beq,
        }

        pol_interp = {
            'a_i': a_i,
            'a_w': a_w,
        }

        return EVa, outputs, pol_interp

    def update_distribution(self, outputs, pol_interp, g0=None, g_newborn=None, **kwargs):
        Pi = self.Pi
        T = self.T
        T_ret = self.T_ret

        nz = len(self.z_grid)
        na = len(self.a_grid)

        gc = kwargs['gc']
        p_death = kwargs['p_death']

        a_i = pol_interp['a_i']
        a_w = pol_interp['a_w']

        g = np.zeros((nz, na, T))

        # agents are born at median income with zero wealth
        if g_newborn is None:
            g_newborn = np.zeros((nz, na))
            assert np.isclose(self.a_grid[0], 0)
            g_newborn[nz // 2, 0] = 1.0
        
        g[..., 0] = g_newborn

        if g0 is not None:
            # initial cohort grows at rate gc
            g[..., 0] *= (1 + gc) * np.sum(g0[..., 0]) / np.sum(g[..., 0])

        for t in range(1, T):
            # print(t)
            if g0 is None:
                g_here = g[..., t - 1] / (1 + gc)
            else:
                g_here = g0[..., t - 1]
            
            if t < T_ret - 1:
                Pi_here = Pi
            else:
                Pi_here = np.eye(nz)

            # first, send each agent at point (z, a) to (z, a')
            g_mid = ha_utils.update_policy_1a(a_i[..., t - 1], a_w[..., t - 1], g_here)

            # then send (z, a') to (z', a')
            g[..., t] = (1 - p_death[t - 1]) * ha_utils.update_exogenous_1a(Pi_here, g_mid)
        
        if g0 is None:
            gp = gc
        else:
            gp = np.sum(g) / np.sum(g0) - 1

        g = g / np.sum(g)

        aggregates = {k.upper(): np.sum(g * outputs[k]) for k in outputs}
        aggregates.update({'gp': gp})

        return g, aggregates
    
    def policy_ss(self, modified_inputs=None):
        inputs = self.inputs.copy()

        if modified_inputs is not None:
            inputs.update(modified_inputs)

        EVa, outputs, pol_interp = self.bellman_iteration(**inputs)

        return EVa, outputs, pol_interp

    def distribution_ss(self, outputs, pol_interp):
        g, aggregates = self.update_distribution(outputs, pol_interp, **self.inputs)
        return g, aggregates

    def steady_state(self):
        EVa, outputs, pol_interp = self.policy_ss()
        g, aggregates = self.distribution_ss(outputs, pol_interp)

        self.EVa = EVa
        self.outputs = outputs
        self.pol_interp = pol_interp

        self.g = g
        self.aggregates = aggregates

        mass_close_to_amax = np.sum(self.g * (self.outputs['a'] > 0.95 * np.max(self.a_grid)))
        if mass_close_to_amax > 0.01:
            print('Too many agents close to amax')

    def mit_shock(self, paths):
        shocked_inputs = list(paths.keys())
        # here T is the truncation horizon for IRFs, not the max age
        T = len(paths[shocked_inputs[0]])

        initial_inputs = self.inputs.copy()
        terminal_inputs = {k: paths[k][-1] for k in paths}

        # check if there is a permanent change in one of the inputs, in which case we need to recompute the steady state
        if not np.all([np.isclose(self.inputs[k], terminal_inputs[k]) for k in shocked_inputs]):
            EVa_next, _, _ = self.policy_ss(terminal_inputs)
        else:
            EVa_next = self.EVa

        inputs_transition = list()
        outputs_transition = list()
        pol_interp_transition = list()

        for t in range(T - 1, -1, -1):
            # print(t)
            inputs_here = initial_inputs.copy()
            inputs_here.update({k: paths[k][t] for k in shocked_inputs})

            EVa_next, outputs_here, pol_interp_here = self.bellman_iteration(EVa_next, **inputs_here)

            inputs_transition.append(inputs_here)
            outputs_transition.append(outputs_here)
            pol_interp_transition.append(pol_interp_here)

        inputs_transition.reverse()
        outputs_transition.reverse()
        pol_interp_transition.reverse()

        aggregate_paths = {k: np.zeros(T) for k in self.aggregates}
        g0 = self.g

        for t in range(T):
            # print(t)
            g0, aggregates_here = self.update_distribution(outputs_transition[t], pol_interp_transition[t], g0=g0, **inputs_transition[t])
            for k in aggregates_here:
                aggregate_paths[k][t] = aggregates_here[k]

        return aggregate_paths


def solve_steady_state(a_grid, z_grid, Pi, crra, r, b_ret, z_age, p_death, gc, T, T_ret, alpha, delta, Y, B_Y, G_Y,
                       A0=1.0, beta0=0.99, w0=1.0, tau0=0.4, transfer0=0.0, solve=True):
    # public debt and spending
    B = B_Y * Y
    G = G_Y * Y

    # initial guess
    x0 = np.array([A0, beta0, w0, tau0, transfer0])

    def fobj(A, beta, w, tau, transfer):
        hh = HouseholdOLG(a_grid, z_grid, Pi, beta, crra, w, tau, r, b_ret, z_age, p_death, gc, transfer, T, T_ret)
        hh.steady_state()

        # capital and labor
        K = hh.aggregates['A_M'] + transfer - B
        L = hh.aggregates['L']

        I = (delta + gc) * K

        Y_implied = A * (K ** alpha) * (L ** (1 - alpha))
        r_implied = A * alpha * ((L / K) ** (1 - alpha)) - delta
        w_implied = A * (1 - alpha) * ((K / L) ** alpha)

        gov_res = G + (r - gc) * B - hh.aggregates['REV']
        transfer_res = (1 + gc) * transfer - hh.aggregates['BEQ']

        budget_res = hh.aggregates['E'] + (r - gc) * (hh.aggregates['A_M'] + transfer) - hh.aggregates['C']
        earnings_res = hh.aggregates['E'] - (w * L - hh.aggregates['REV'])
        goods_mkt = Y - (hh.aggregates['C'] + I + G)
        asset_res = hh.aggregates['A'] - (1 + gc) * (hh.aggregates['A_M'] + transfer)

        diff = np.array([
            Y - Y_implied,
            r - r_implied,
            w - w_implied,
            gov_res,
            transfer_res,
        ])

        out = {
            'A': A,
            'Y': Y,
            'K': K,
            'L': L,
            'B': B,
            'K': K,
            'I': I,
            'tau': tau,
            'w': w,
            'beta': beta,
            'goods_mkt': goods_mkt,
            'budget_res': budget_res,
            'earnings_res': earnings_res,
            'asset_res': asset_res,
        }

        return diff, hh, out
    
    if solve:
        x = opt.fsolve(lambda x: fobj(*x)[0], x0)
    else:
        x = x0

    return fobj(*x)

