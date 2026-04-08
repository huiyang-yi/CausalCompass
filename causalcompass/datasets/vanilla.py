# Portions of this file are adapted from Neural-GC
# Copyright (c) 2019 Ian Covert
# https://github.com/iancovert/Neural-GC
# Licensed under the MIT License

"""
Data generation for VAR and Lorenz 96 datasets.

Reference:
    [1] https://github.com/iancovert/Neural-GC/blob/master/synthetic.py
"""

import numpy as np
from scipy.integrate import odeint
import os

def make_var_stationary(beta, radius=0.97):
    '''Rescale coefficients of VAR model to make stable.'''
    p = beta.shape[0]
    lag = beta.shape[1] // p
    bottom = np.hstack((np.eye(p * (lag - 1)), np.zeros((p * (lag - 1), p))))
    beta_tilde = np.vstack((beta, bottom))
    eigvals = np.linalg.eigvals(beta_tilde)
    max_eig = max(np.abs(eigvals))
    nonstationary = max_eig > radius
    if nonstationary:
        return make_var_stationary(0.95 * beta, radius)
    else:
        return beta


def simulate_var(p, T, lag=3, sparsity=0.2, beta_value=1.0, sd=0.1, burn_in=100, seed=0):
    """
    Generate time series data from a Vector Autoregressive (VAR) model.

    References
    ----------
    https://github.com/iancovert/Neural-GC

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
    burn_in : int, default 100
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, beta, GC) — time series array of shape (T, p), coefficient matrix, and ground-truth causal graph of shape (p, p).
    """
    if seed is not None:
        np.random.seed(seed)

    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta[i, choice] = beta_value
        GC[i, choice] = 1

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    # Generate data.
    errors = np.random.normal(scale=sd, size=(p, T + burn_in))
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
        X[:, t] += + errors[:, t - 1]

    return X.T[burn_in:], beta, GC


def lorenz(x, t, F):
    '''Partial derivatives for Lorenz-96 ODE.'''
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F

    return dxdt


def simulate_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000,
                       seed=0):
    """
    Generate time series data from the Lorenz-96 dynamical system.

    References
    ----------
    https://github.com/iancovert/Neural-GC

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
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, GC) — time series array of shape (T, p) and ground-truth causal graph of shape (p, p).
    """
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC
