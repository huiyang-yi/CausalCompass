import numpy as np
import os
from typing import Union, List
from .vanilla import simulate_var, simulate_lorenz_96

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


def simulate_var_with_measure_error(p, T, lag=3, gamma=1.2, sparsity=0.2, beta_value=1.0, sd=0.1, burn_in=100, seed=0):
    """
    Generate VAR data with measurement error proportional to data variance.

    Parameters
    ----------
    p : int
        Number of variables
    T : int
        Number of time points
    lag : int, default 3
        Number of lags in the VAR model
    gamma : float, default 1.2
        Scale factor for measurement error variance
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
    X_clean, beta, GC = simulate_var(
        p=p, T=T, lag=lag,
        sparsity=sparsity, beta_value=beta_value, sd=sd, burn_in=burn_in, seed=seed
    )
    X_meas = add_measure_error(X_clean, gamma=gamma)
    return X_meas, beta, GC


def simulate_lorenz_with_measure_error(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000, gamma=1.2, seed=0):
    """
    Generate Lorenz-96 data with measurement error.

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
    sd : float, default 0.1
        Noise standard deviation
    burn_in : int, default 1000
        Burn-in period
    gamma : float, default 1.2
        Scale factor for measurement error variance
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, GC) — time series array of shape (T, p) and ground-truth causal graph of shape (p, p).
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
