import numpy as np
import os
from scipy.integrate import odeint
from .vanilla import make_var_stationary, lorenz

def simulate_var_with_confounders(p, T, lag=3, rho=0.5, sparsity=0.2,
                                  beta_value=1.0, sd=0.1, burn_in=100, seed=0):
    """
    Generate VAR data with cross-lag hidden confounders that introduce spurious correlations.

    Parameters
    ----------
    p : int
        Number of variables
    T : int
        Number of time points
    lag : int, default 3
        Number of lags in the VAR model
    rho : float, default 0.5
        Probability of confounding between each pair of variables
    sparsity : float, default 0.2
        Sparsity of the causal graph
    beta_value : float, default 1.0
        Coefficient value
    sd : float, default 0.1
        Noise standard deviation
    burn_in : int, default 100
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, beta, GC) — time series array of shape (T, p), coefficient matrix, and ground-truth causal graph of shape (p, p)
    """
    if seed is not None:
        np.random.seed(seed)

    total_vars = 2 * p  # p confounders + p observed variables

    # Set up coefficients and Granger causality ground truth for observed variables
    GC = np.eye(p, dtype=int)
    beta_obs = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta_obs[i, choice] = beta_value
        GC[i, choice] = 1

    # Create cross-lag confounder structure
    C0 = np.zeros((p, p))
    C_lags = [np.zeros((p, p)) for _ in range(lag)]

    for i in range(p):  # target variable
        for j in range(p):  # source variable
            if i == j or GC[i, j] == 1:
                continue

            if np.random.binomial(1, rho) == 1:
                k = np.random.choice(p)  # choose confounder Uk

                C0[j, k] = beta_value

                for tau in range(lag):
                    C_lags[tau][i, k] = beta_value

    # Construct multi-lag coefficient matrix
    blocks = []
    for tau in range(lag):
        B = np.zeros((total_vars, total_vars))

        B[:p, :p] = np.eye(p) * beta_value

        # Observed variable dynamics
        B[p:, p:] = beta_obs

        # Cross-lag confounder effects
        B[p:, :p] = C_lags[tau]

        blocks.append(B)

    beta = np.hstack(blocks)
    beta = make_var_stationary(beta)

    # Generate data
    errors = np.random.normal(scale=sd, size=(total_vars, T + burn_in))
    X = np.zeros((total_vars, T + burn_in))
    X[:, :lag] = errors[:, :lag]

    for t in range(lag, T + burn_in):
        # VAR dynamics
        lagged = np.dot(beta, X[:, (t - lag):t].flatten(order='F')) + errors[:, t - 1]

        # Split confounders and observed variables
        U_t = lagged[:p]
        X_t = lagged[p:]

        # Add contemporaneous confounder effects
        X_t += np.dot(C0, U_t)

        # Update state
        X[:p, t] = U_t
        X[p:, t] = X_t

    # Extract observed variables and remove burn-in
    X_observed = X[p:, burn_in:].T

    return X_observed, beta, GC


def simulate_lorenz_with_confounders(p, T, F=10.0, delta_t=0.1, rho=0.5,
                                     sd=0.1, burn_in=1000, seed=0):
    """
    Generate Lorenz-96 data with hidden confounders.

    Parameters
    ----------
    p : int
        Number of variables
    T : int
        Number of time points
    F : float, default 10.0
        Forcing parameter
    delta_t : float, default 0.1
        Time step for ODE solver
    rho : float, default 0.5
        Probability of confounding between each pair of variables
    sd : float, default 0.1
        Noise standard deviation
    burn_in : int, default 1000
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, GC) — time series array of shape (T, p) and ground-truth causal graph of shape (p, p).
    """
    if seed is not None:
        np.random.seed(seed)

    total_vars = 2 * p
    total_T = T + burn_in

    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    confounder_structure = np.zeros((p, p))
    for i in range(p):
        for j in range(p):
            if i != j and (GC[i, j] != 1 or GC[j, i] != 1):
                if np.random.binomial(1, rho) == 1:
                    k = np.random.choice(p)
                    confounder_structure[k, i] = 1
                    confounder_structure[k, j] = 1

    # Extended Lorenz system with confounders
    def lorenz_with_confounders(x, t, F):
        dxdt = np.zeros(total_vars)

        # Confounder dynamics: first p variables (Lorenz-96)
        for i in range(p):
            dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F

        # Observed variable dynamics: second p variables (Lorenz-96 + confounder influences)
        for i in range(p, total_vars):
            j = i - p
            next_idx = (j + 1) % p + p
            prev_idx = (j - 1) % p + p
            prev2_idx = (j - 2) % p + p

            # Standard Lorenz-96 dynamics
            lorenz_term = (x[next_idx] - x[prev2_idx]) * x[prev_idx] - x[i] + F

            # Add confounder influences
            confounder_influence = 0
            for k in range(p):
                if confounder_structure[k, j] == 1:
                    confounder_influence += 3.0 * x[k]  # confounder effect

            dxdt[i] = lorenz_term + confounder_influence

        return dxdt

    # Initial conditions and ODE solving
    x0 = np.random.normal(scale=0.01, size=total_vars)
    t = np.linspace(0, total_T * delta_t, total_T)
    X = odeint(lorenz_with_confounders, x0, t, args=(F,))

    # Add noise
    X += np.random.normal(scale=sd, size=(total_T, total_vars))

    # Extract observed variables and remove burn-in
    X_observed = X[burn_in:, p:]

    return X_observed, GC
