"""
Standardized data generation for VAR and Lorenz-96 systems.

This module generates normalized datasets using:
- Z-score normalization (zero mean, unit variance)
- Min-max normalization (scaled to [0, 1])

Both methods are applied to vanilla VAR and Lorenz-96 time series data.
"""

import numpy as np
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .vanilla import simulate_var, simulate_lorenz_96


def normalize_data(X_np, method):
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'")
    return scaler.fit_transform(X_np)


def generate_standardized_var(p, T, lag=3, sparsity=0.2, beta_value=1.0, sd=0.1, method='zscore', burn_in=100, seed=0):
    """
    Generate VAR data with z-score or min-max normalization applied.

    Parameters
    ----------
    p : int
        Number of variables
    T : int
        Number of time points
    lag : int, default 3
        Number of lags in the VAR model
    sparsity : float, default 0.2
        Sparsity of the causal graph
    beta_value : float, default 1.0
        Coefficient value
    sd : float, default 0.1
        Noise standard deviation
    method : str, default 'zscore'
        Normalization method: 'zscore' or 'minmax'
    burn_in : int, default 100
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, beta, GC) — time series array of shape (T, p), coefficient matrix, and ground-truth causal graph of shape (p, p).
    """
    X_np, beta, GC = simulate_var(p, T, lag=lag, sparsity=sparsity, beta_value=beta_value, sd=sd, burn_in=burn_in, seed=seed)
    X_scaled = normalize_data(X_np, method)
    return X_scaled.astype(np.float32), beta, GC


def generate_standardized_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, method='zscore', burn_in=1000, seed=0):
    """
    Generate Lorenz-96 data with normalization applied.

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
    method : str, default 'zscore'
        Normalization method: 'zscore' or 'minmax'
    burn_in : int, default 1000
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, GC) — time series array of shape (T, p) and ground-truth causal graph of shape (p, p).
    """
    X_np, GC = simulate_lorenz_96(p, T, F=F, delta_t=delta_t, sd=sd, burn_in=burn_in, seed=seed)
    X_scaled = normalize_data(X_np, method)
    return X_scaled.astype(np.float32), GC
