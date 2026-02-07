"""
Confounder data generation for VAR and Lorenz-96 systems.

This module generates time series data with hidden confounders that create
spurious correlations between observed variables.

"""
import numpy as np
import os
from scipy.integrate import odeint
from vanilla import make_var_stationary, lorenz


def simulate_var_with_confounders(p, T, lag=3, rho=0.2, sparsity=0.2,
                                  beta_value=1.0, sd=0.1, seed=0):
    """
    Generate VAR data with cross-lag confounders.

    Parameters
    ----------
    p : int
        Number of observed variables
    T : int
        Number of time points
    lag : int
        Number of lags in VAR model
    rho : float
        Probability of confounding between each pair of variables
    sparsity : float
        Sparsity of connections between observed variables
    beta_value : float
        Magnitude of VAR and confounder coefficients
    sd : float
        Standard deviation of noise terms
    seed : int
        Random seed

    Returns
    -------
    X_observed : np.ndarray, shape (T, p)
        Time series data for observed variables only
    beta : np.ndarray
        VAR coefficients matrix for the full system
    GC : np.ndarray, shape (p, p)
        Granger causality ground truth for observed variables
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
    C0 = np.zeros((p, p))  # contemporaneous confounder effects
    C_lags = [np.zeros((p, p)) for _ in range(lag)]  # lagged confounder effects

    for i in range(p):  # target variable
        for j in range(p):  # source variable
            if i == j or GC[i, j] == 1:  # skip self-loops and existing edges
                continue

            if np.random.binomial(1, rho) == 1:
                k = np.random.choice(p)  # choose confounder Uk

                # Contemporaneous: Uk(t) -> Xj(t)
                C0[j, k] = beta_value

                # Cross-lag: Uk(t-τ) -> Xi(t) for all τ=1..lag
                for tau in range(lag):
                    C_lags[tau][i, k] = beta_value

    # Construct multi-lag coefficient matrix
    blocks = []
    for tau in range(lag):
        B = np.zeros((total_vars, total_vars))

        # Confounder dynamics (diagonal VAR)
        B[:p, :p] = np.eye(p) * beta_value

        # Observed variable dynamics
        B[p:, p:] = beta_obs

        # Cross-lag confounder effects
        B[p:, :p] = C_lags[tau]

        blocks.append(B)

    beta = np.hstack(blocks)
    beta = make_var_stationary(beta)

    # Generate data
    burn_in = 100
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


def simulate_lorenz_with_confounders(p, T, F=10.0, delta_t=0.1, rho=0.2,
                                     sd=0.1, burn_in=1000, seed=0):
    """
    Generate Lorenz-96 data with cross-lag confounders.

    Parameters
    ----------
    p : int
        Number of observed variables
    T : int
        Number of time points
    F : float
        Forcing parameter for Lorenz-96
    delta_t : float
        Time step for ODE integration
    rho : float
        Probability of confounding between each pair of variables
    sd : float
        Standard deviation of noise terms
    burn_in : int
        Number of burn-in time steps
    seed : int
        Random seed

    Returns
    -------
    X_observed : np.ndarray, shape (T, p)
        Time series data for observed variables only
    GC : np.ndarray, shape (p, p)
        Granger causality ground truth for observed variables
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
            j = i - p  # index in observed system
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


def save_data(data, beta_or_gc, dataset_type, p, T, seed, F=None, gc=None, rho=0.5):
    """Save confounder data with consistent naming convention."""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "confounder")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == 'VAR':
        filename = f'confounder_rho{rho}_VAR_p{p}_T{T}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, beta=beta_or_gc, gc=gc)
    elif dataset_type == 'Lorenz':
        filename = f'confounder_rho{rho}_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, gc=beta_or_gc)


def load_and_check_data(ps, Ts, Fs, rho=0.5):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "confounder")

    print("\n--- Checking confounder VAR data ---")
    for p in ps:
        for T in Ts:
            filename = os.path.join(output_dir, f"confounder_rho{rho}_VAR_p{p}_T{T}_seed0.npz")
            if os.path.exists(filename):
                data = np.load(filename)
                print(f"Loaded {filename}")
                print(
                    f"data shape: {data['data'].shape}, beta shape: {data['beta'].shape}, gc shape: {data['gc'].shape}")
            else:
                print(f"{filename} not found!")

    print("\n--- Checking confounder Lorenz data ---")
    for p in ps:
        for T in Ts:
            for F in Fs:
                filename = os.path.join(output_dir, f"confounder_rho{rho}_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
                if os.path.exists(filename):
                    data = np.load(filename)
                    print(f"Loaded {filename}")
                    print(f"data shape: {data['data'].shape}, gc shape: {data['gc'].shape}")
                else:
                    print(f"{filename} not found!")


if __name__ == "__main__":
    ps = [10]
    Ts = [500, 1000]
    Fs = [10, 40]
    seeds = range(5)
    lag = 3
    rho = 0.5  # Probability of confounding

    # Generate confounder VAR data
    for p in ps:
        for T in Ts:
            for seed in seeds:
                print(f"Generating confounder VAR: p={p}, T={T}, rho={rho}, seed={seed}")
                data, beta, gc = simulate_var_with_confounders(
                    p=p, T=T, lag=lag, rho=rho, seed=seed
                )
                save_data(data, beta, dataset_type='VAR', p=p, T=T, seed=seed, gc=gc, rho=rho)

    # Generate confounder Lorenz data
    for p in ps:
        for T in Ts:
            for F in Fs:
                for seed in seeds:
                    print(f"Generating confounder Lorenz: p={p}, T={T}, F={F}, rho={rho}, seed={seed}")
                    data, gc = simulate_lorenz_with_confounders(
                        p=p, T=T, F=F, rho=rho, seed=seed
                    )
                    save_data(data, gc, dataset_type='Lorenz', p=p, T=T, F=F, seed=seed, rho=rho)

    # Load and check generated data
    load_and_check_data(ps, Ts, Fs, rho=rho)
