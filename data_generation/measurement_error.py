"""
Measurement error data generation for VAR and Lorenz-96 systems.


"""

import numpy as np
import os
from typing import Union, List
from vanilla import simulate_var, simulate_lorenz_96


def add_measure_error(X: np.ndarray, gamma: float) -> np.ndarray:
    """
    Add measurement error with variance proportional to data variance.

    Measurement error ~ N(0, gamma * Var(X))

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
        Clean time-series data (time along axis 0).
    gamma : float
        Scale factor for measurement error variance.
        err_var = gamma * Var(X), where Var(X) is computed per variable.
        e.g., gamma=1.2 means measurement error variance is 1.2 times the data variance.

    Returns
    -------
    X_meas : np.ndarray, shape (T, p)
        Data with measurement error added.

    Notes
    -----
    - Variance is computed per variable (column-wise)
    - Each variable gets independent Gaussian noise with variance = gamma * var(X_i)
    """
    T, p = X.shape

    # Compute variance for each variable (column-wise)
    X_var = np.var(X, axis=0)  # shape: (p,)

    # Measurement error variance for each variable
    err_var = gamma * X_var  # shape: (p,)

    # Standard deviation for each variable
    err_std = np.sqrt(err_var)  # shape: (p,)

    # Generate independent Gaussian noise for each variable
    # Each column uses different std based on that variable's data variance
    err = np.random.standard_normal(size=(T, p)) * err_std[np.newaxis, :]  # broadcasting

    return X + err


def simulate_var_with_measure_error(p, T, lag, gamma=1.2,
                                    sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    """
    Generate VAR data with measurement error.

    Parameters:
    -----------
    p : int
        Number of variables
    T : int
        Number of time steps
    lag : int
        Lag order
    gamma : float
        Measurement error scale factor
        err_var = gamma * Var(X) per variable
    sparsity : float
        Sparsity of causal graph
    beta_value : float
        Magnitude of causal effects
    sd : float
        Standard deviation of process noise
    seed : int
        Random seed

    Returns:
    --------
    X_meas : np.ndarray, shape (T, p)
        Data with measurement error
    beta : np.ndarray
        Coefficient matrix
    GC : np.ndarray
        Ground truth causal graph
    """
    X_clean, beta, GC = simulate_var(
        p=p, T=T, lag=lag,
        sparsity=sparsity, beta_value=beta_value, sd=sd, seed=seed
    )
    X_meas = add_measure_error(X_clean, gamma=gamma)
    return X_meas, beta, GC


def simulate_lorenz_with_measure_error(p, T, F=10.0, delta_t=0.1, sd=0.1,
                                       burn_in=1000, gamma=1.2, seed=0):
    """
    Generate Lorenz-96 data with measurement error.

    Parameters:
    -----------
    p : int
        Number of variables
    T : int
        Number of time steps
    F : float
        Forcing parameter for Lorenz-96
    delta_t : float
        Time step size
    sd : float
        Standard deviation of process noise
    burn_in : int
        Number of initial steps to discard
    gamma : float
        Measurement error scale factor
        err_var = gamma * Var(X) per variable
    seed : int
        Random seed

    Returns:
    --------
    X_meas : np.ndarray, shape (T, p)
        Data with measurement error
    GC : np.ndarray
        Ground truth causal graph
    """
    X_clean, GC = simulate_lorenz_96(
        p=p, T=T, F=F, delta_t=delta_t, sd=sd, burn_in=burn_in, seed=seed
    )
    X_meas = add_measure_error(X_clean, gamma=gamma)
    return X_meas, GC


def _gamma_to_str(gamma: Union[float, List[float]]) -> str:
    """Convert gamma parameter to string for filename."""
    if isinstance(gamma, list):
        return "list"
    else:
        return f"{gamma}"


def save_me_data(data, beta_or_gc, dataset_type, p, T, seed, F=None, gc=None, gamma=1.2):
    """Save measurement error data with consistent naming convention."""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "measurement_error")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == 'VAR':
        filename = f'ME_gamma{gamma}_VAR_p{p}_T{T}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, beta=beta_or_gc, gc=gc)
    elif dataset_type == 'Lorenz':
        filename = f'ME_gamma{gamma}_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, gc=beta_or_gc)


def load_and_check_me_data(ps, Ts, Fs, gamma):
    """Load and verify saved measurement error data."""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(parent_dir, "datasets", "measurement_error")

    print("\n--- Checking saved ME VAR data ---")
    for p in ps:
        for T in Ts:
            fname = os.path.join(out_dir, f"ME_gamma{gamma}_VAR_p{p}_T{T}_seed0.npz")
            if os.path.exists(fname):
                data = np.load(fname)
                print(f"Loaded {fname}")
                print(f"  data(meas) shape: {data['data'].shape}")
                print(f"  beta shape: {data['beta'].shape}")
                print(f"  gc shape: {data['gc'].shape}")

            else:
                print(f"{fname} not found!")

    print("\n--- Checking saved ME Lorenz data ---")
    for p in ps:
        for T in Ts:
            for F in Fs:
                fname = os.path.join(out_dir, f"ME_gamma{gamma}_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
                if os.path.exists(fname):
                    data = np.load(fname)
                    print(f"Loaded {fname}")
                    print(f"  data(meas) shape: {data['data'].shape}")
                    print(f"  gc shape: {data['gc'].shape}")

                else:
                    print(f"{fname} not found!")


if __name__ == "__main__":
    ps = [10]  # [10, 15]
    Ts = [500, 1000]
    Fs = [10, 40]
    seeds = range(5)
    lag = 3
    gamma = 1.2  # Measurement error variance = 1.2 * data variance

    print(f"Generating measurement error datasets with gamma={gamma}")
    print(f"Measurement error: err ~ N(0, {gamma} * Var(X))")
    print("=" * 60)

    # Generate VAR data
    for p in ps:
        for T in Ts:
            for seed in seeds:
                print(f"Generating ME VAR: p={p}, T={T}, gamma={gamma}, seed={seed}")
                X_meas, beta, gc = simulate_var_with_measure_error(
                    p=p, T=T, lag=lag, gamma=gamma, seed=seed
                )
                save_me_data(X_meas, beta_or_gc=beta,
                             dataset_type='VAR', p=p, T=T, seed=seed, gc=gc, gamma=gamma)

    # Generate Lorenz data
    for p in ps:
        for T in Ts:
            for F in Fs:
                for seed in seeds:
                    print(f"Generating ME Lorenz: p={p}, T={T}, F={F}, gamma={gamma}, seed={seed}")
                    X_meas, gc = simulate_lorenz_with_measure_error(
                        p=p, T=T, F=F, gamma=gamma, seed=seed
                    )
                    save_me_data(X_meas, beta_or_gc=gc,
                                 dataset_type='Lorenz', p=p, T=T, F=F, seed=seed, gamma=gamma)

    # Load and verify data
    load_and_check_me_data(ps, Ts, Fs, gamma)
