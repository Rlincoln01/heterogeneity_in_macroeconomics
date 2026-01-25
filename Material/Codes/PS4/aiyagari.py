"""
##############################################
#              Aiyagari Model                #
##############################################

ile with all the classes that compose the building blocks of the Aiyagari Model. It has two classes,
Households and Firms, where I compute in the latter the policy functions, joint stationary distribution of income and wealth,
and aggregate household variables (capital, consumption, savings, etc).
The function steady_state in the Households class computes the steady state (in partial equilibrium) given a level of wages and interest rate. 

Functions:
- solve_steady_state: Function uses built-in fsolve to find the steady state interest rate of the Aiyagari model.

Classes:
- Household: All the functions related to the household's decision-making process.
- Firm: All the functions related to the firm's decision-making process.


Author: Rafael Lincoln 

Date: 24 Apr 2025 (of creation)
last update: 26 Apr 2025
"""

# Import Libraries
import numpy as np
import ha_utils 
import scipy.optimize as opt


# 1. Household Class ----------------------------------------------------

class Households:
    """
    Class representing a household in the Aiyagari model.
    """

    def __init__(self, a_grid, z_grid, Pi, beta, r, w, gamma):
        """
        Initialize the Household class.

        Parameters:
        - a_grid: Asset grid.
        - z_grid: Income grid.
        - Pi: Transition matrix for income.
        - beta: Discount factor.
        - rho: Persistence of income process.
        - sigma_e: Standard deviation of income shocks.
        """
        self.a_grid = a_grid
        self.z_grid = z_grid
        self.Pi = Pi
        self.beta = beta
        self.r = r
        self.w = w
        self.gamma = gamma

        # preallocation for the equilibrium objects of the class
        self.pol_functions = None
        self.pol_inter = None
        self.g = None
        self.aggregates = None
        
    def EGM(self,params,show_iter = True):
        """
        Solve the Income Fluctuation Problem by the Endogenous Gridpoint Method.

        Params:
        ------
        dict with
            - tol      : convergence tolerance for c
            - max_iter : maximum number of EGM steps

        Returns:
        -------            
        """
        beta, w, R, gamma    = self.beta, self.w, 1+self.r, self.gamma
        A, Z, P              = self.a_grid, self.z_grid, self.Pi
        na, nz               = len(A), len(Z)
        amin                 = A[0]


        # 1st step: Guess for c(a,z)
        z_mesh, a_mesh = np.meshgrid(Z, A, indexing='ij')
        budget = R*a_mesh + w*np.exp(z_mesh) # Budget constraint
        c_old = budget.copy() # Initial guess for consumption
        # Preallocation
        c_new = np.empty_like(c_old) # New Consumption Policy Function
        a = np.empty_like(c_old) # Next Period asset policy function
        a_i = np.empty_like(c_old) # Array to store next period's asset grid points indexes
        a_w = np.empty_like(c_old) # Array to store interpolation weights

        for it in range(params["max_iter"]):
            # Preallocation for the endogenous grid and c_foc
            c_foc  = np.empty_like(c_old)
            # 2nd Step: Calculate c_FOC and a_star
            for ih in range(nz):
                for ia in range(na):
                    a_prime = A[ia] # Tomorrow's asset grid point
                    # 2.1 - Calculate the Expected Marginal Utility of Consumption
                    emuc = 0.0
                    for jp in range(nz): # iterating again over z_grid
                        emuc += P[ih,jp]*(c_old[jp,ia]**(-gamma)) # Expected Marginal Utility of Consumption
                    # calculate C_foc
                    c_foc[ih,ia] = (beta*R*emuc)**(-1/gamma)

            # 3rd step: Update the consumption policy function, and find asset policy function
            c_new, a, a_i, a_w = ha_utils.solve_euler_one_asset(c_foc, budget, A)

            # 4th Step: Convergence Check
            diff = np.max(np.abs(c_new-c_old))
            if it % 50 == 0 and show_iter == True:
                print(f"EGM iter {it}: max policy function change = {diff}")

            if diff < params["tol"]:
                print(f"EGM converged in {it} iters, Δ={diff:.2e}")
                break
            c_old = c_new.copy() # Update for next iteration
        else:
            print("EGM failed to converge")

        # Final Policy Functions:

        outputs = {
            "a": a,
            "c": c_new,
            "l": np.exp(z_mesh),
            "sav": a - A[np.newaxis,:],
        }

        pol_interp = {
            'a_i': a_i,
            'a_w': a_w,
        }

        return outputs, pol_interp
    
    
    def policy_ss(self, modified_inputs =None ):
        """
        Compute the policy functions in steady state.

        Parameters:
        - modified_inputs: Optional dictionary to modify inputs.

        Returns:
        - Policy functions in steady state.
        """
        if modified_inputs is not None:
            for key, value in modified_inputs.items():
                setattr(self, key, value)

        # Call EGM method to compute policy functions
        pol_functions, pol_inter = self.EGM(params={'tol': 1e-6, 'max_iter': 1000},show_iter=False)

        return pol_functions, pol_inter

    def stationary_dist(self):
        """
        Compute the stationary distribution of wealth.

        Returns:
        - Stationary distribution of wealth.
        """
        # Compute the policy functions 
        _, pol_inter = self.policy_ss()

        # Transition Matrix
        T = ha_utils.compute_joint_transition_matrix(pol_inter['a_i'],pol_inter['a_w'],self.Pi)
        
        # Compute stationary distribution
        stationary_dist = ha_utils.stationary_dist(T)
        stationary_dist = stationary_dist.reshape(len(self.z_grid), len(self.a_grid))
        
        return stationary_dist


    def aggregate_vars(self):
        """
        Compute aggregate variables.

        Returns:
        - Aggregate variables.
        """
        # Compute the stationary distribution & aggregates
        g = self.stationary_dist()
        outputs,_ = self.EGM(params={'tol': 1e-6, 'max_iter': 1000})
        
        # Compute Aggregate Variables
        aggregates = {k.upper(): np.sum(g * outputs[k]) for k in outputs}        

        return aggregates
    
    def steady_state(self):
        """
        Compute the stationary equilibrium.

        Returns:
        - Stationary equilibrium values.
        """
        # Policy Functions
        pol_functions, pol_inter = self.policy_ss()
        # Compute the stationary distribution
        T = ha_utils.compute_joint_transition_matrix(pol_inter['a_i'], pol_inter['a_w'], self.Pi)
        g = ha_utils.stationary_dist(T).reshape(len(self.z_grid), len(self.a_grid))
        # Compute aggregate variables
        aggregates = {k.upper(): np.sum(g * pol_functions[k]) for k in pol_functions}        

        self.pol_functions = pol_functions
        self.pol_inter = pol_inter
        self.g = g
        self.aggregates = aggregates




class Firms:
    """
    Class representing a firm in the Aiyagari model.
    """

    def __init__(self, alpha, delta,r,Z = 1):
        """
        Initialize the Firm class.

        Parameters:
        - alpha: Capital share in output.
        - delta: Depreciation rate.
        - r: Interest rate.
        """
        self.alpha = alpha
        self.delta = delta
        self.r = r
        self.Z = Z
        
        
    def output(self, K, L):
        """
        Calculate the production function.

        Parameters:
        - Z: Technology level.
        - K: Capital stock.
        - L: Labor input.

        Returns:
        - Output produced by the firm.
        """
        return self.Z*K**self.alpha * L**(1-self.alpha)
    
    
    def eq_wage(self):
        """
        Calculate the equilibrium wage from the interest rate and parameters of the production function.

        Returns:
        - Wage rate.
        """
        alpha, delta, r = self.alpha, self.delta, self.r
        Z = self.Z

        return (1-alpha)*(Z**(1/(1-alpha)))*(alpha/(r+delta))**(alpha/(1-alpha))
    
    def capital_demand(self, L):
        """
        Calculate the capital demand given the capital stock and labor input.


        Returns:
        - Capital demand.
        """
        alpha = self.alpha
        Z = self.Z
        delta = self.delta
        r = self.r

        K_d = L*(alpha*Z/(r+delta))**(1/(1-alpha))
        return K_d
    

def solve_steady_state(a_grid, z_grid, Pi, alpha, delta, beta, gamma,
                       r0 =0.04,solve = True):
    """
    Solve the steady state of the Aiyagari model.

    Parameters:
    - alpha: Capital share in output.
    - delta: Depreciation rate.
    - beta: Discount factor.
    - gamma: Coefficient of relative risk aversion.
    - r: Interest rate.
    - w: Wage rate.

    Returns:
    - Steady state values.
    """
    
    # Initial Guess for the interest rate at the steady state
    x0 = np.array([r0])

    # Define Objective Function
    def fobj(r):
        # Initialize the Household and Firm classes
        ff = Firms(alpha, delta, r)
        w = ff.eq_wage()
        hh = Households(a_grid, z_grid, Pi, beta, r, w, gamma)


        # Compute the Supply and Demand of Capital
        hh.steady_state()
        aggregates = hh.aggregates
        K_s =  aggregates['A']
        L = aggregates['L']
        K_d = ff.capital_demand(L)

        # Compute the difference between supply and demand
        diff = np.array([K_s - K_d])

        out = {
            "Y": ff.output(aggregates['A'], aggregates['L']),
            "K": aggregates['A'],
            "L": aggregates['L'],
            "C": aggregates['C'],
            "I": ff.output(aggregates['A'], aggregates['L']) - aggregates['C'], 
            "w": w,
            "r": r
        }
    
        return diff, hh, out
    
    if solve:
        x = opt.fsolve(lambda x: fobj(*x)[0], x0)
    else:
        x = x0

    return fobj(*x)