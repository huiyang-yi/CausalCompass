"""
Nonstationary data generation for VAR and Lorenz-96 systems.

This module supports:
1. Time-varying noise variance (VAR+Lorenz-96)
2. Time-varying coefficients (VAR only)

Both variations use Gaussian Process to generate smooth temporal changes.
"""

import numpy as np
from scipy.integrate import odeint
from .vanilla import make_var_stationary
import os
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel


def rbf_kernel(t, l):
    t = np.array(t).reshape(-1, 1)
    return sklearn_rbf_kernel(t, gamma=1 / (2 * l ** 2))


def simulate_nonstationary_var_timevarying_coef(p, T, lag=3, sparsity=0.2, sd=0.1,
                                                beta_value=1.0, noise_std=1.0,
                                                mean_log_sigma=1.0,
                                                coef_noise_std=0.3,
                                                burn_in=100,
                                                seed=0):
    """
    Generate VAR data with both time-varying coefficients and time-varying noise.

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
    sd : float, default 0.1
        Noise standard deviation
    beta_value : float, default 1.0
        Coefficient value
    noise_std : float, default 1.0
        Standard deviation of the GP
    mean_log_sigma : float, default 1.0
        Mean of the GP
    coef_noise_std : float, default 0.3
        Standard deviation of coefficient perturbation
    burn_in : int, default 100
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, beta_t, sigma_t, GC, beta_base)— time series array of shape (T, p), time-varying VAR coefficient matrices of shape (T, p, p*lag), time-varying noise scaling factor of shape (T, p), ground-truth causal graph of shape (p, p), and base (stationary) VAR coefficient matrix before time-varying perturbation.
    """
    if seed is not None:
        np.random.seed(seed)

    GC = np.eye(p, dtype=int)
    beta_base = np.eye(p) * beta_value
    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        GC[i, choice] = 1
        beta_base[i, choice] = beta_value

    beta_base = np.hstack([beta_base for _ in range(lag)])
    beta_base = make_var_stationary(beta_base)

    total_T = T + burn_in

    time = np.linspace(0, 1, total_T)
    K = rbf_kernel(time, l=0.2)
    K += 1e-4 * np.eye(len(K))

    beta_t = np.zeros((total_T, p, p * lag))  # shape: (T, p, p*lag)

    for i in range(p):
        for j in range(p * lag):
            if beta_base[i, j] != 0:
                b_ij_t = np.random.multivariate_normal(
                    mean=np.ones(total_T) * beta_base[i, j],
                    cov=(coef_noise_std ** 2) * K
                )
                beta_t[:, i, j] = b_ij_t
            else:
                beta_t[:, i, j] = 0.0

    for t in range(total_T):
        beta_t[t] = make_var_stationary(beta_t[t])

    sigma_t = np.zeros((total_T, p))
    log_sigma_t = np.random.multivariate_normal(
        mean=np.ones(total_T) * mean_log_sigma,
        cov=(noise_std ** 2) * K
    )
    sigma_t_shared = np.exp(log_sigma_t)
    sigma_t = np.tile(sigma_t_shared[:, None], (1, p))

    X = np.zeros((p, total_T))
    errors = np.random.normal(loc=0.0, scale=sd, size=(p, total_T))
    X[:, :lag] = errors[:, :lag] * sigma_t[:lag].T

    for t in range(lag, total_T):
        phi = np.dot(beta_t[t], X[:, (t - lag):t].flatten(order='F'))
        X[:, t] = phi + sigma_t[t] * errors[:, t]

    return X.T[burn_in:], beta_t[burn_in:], sigma_t[burn_in:], GC, beta_base


def simulate_nonstationary_var(p, T, lag=3, sparsity=0.2, sd=0.1,
                               beta_value=1.0, noise_std=1.0, mean_log_sigma=1.0, burn_in=100, seed=0):
    """
    Generate VAR data with time-varying noise variance (driven by Gaussian Process).

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
    sd : float, default 0.1
        Noise standard deviation
    beta_value : float, default 1.0
        Coefficient value
    noise_std : float, default 1.0
        Standard deviation of the GP
    mean_log_sigma : float, default 1.0
        Mean of the GP
    burn_in : int, default 100
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, beta, sigma_t, GC) — time series array of shape (T, p), VAR coefficient matrix, time-varying noise scaling factor of shape (T, p), and ground-truth causal graph of shape (p, p).
    """
    if seed is not None:
        np.random.seed(seed)

    GC = np.eye(p, dtype=int)
    beta = np.eye(p) * beta_value
    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        GC[i, choice] = 1
        beta[i, choice] = beta_value

    beta = np.hstack([beta for _ in range(lag)])
    beta = make_var_stationary(beta)

    total_T = T + burn_in

    time = np.linspace(0, 1, total_T)
    K = rbf_kernel(time, l=0.2)
    K += 1e-4 * np.eye(len(K))  # Numerical stability
    sigma_t = np.zeros((total_T, p))
    log_sigma_t = np.random.multivariate_normal(
        mean=np.ones(total_T) * mean_log_sigma,
        cov=(noise_std ** 2) * K
    )
    sigma_t_shared = np.exp(log_sigma_t)
    sigma_t = np.tile(sigma_t_shared[:, None], (1, p))

    X = np.zeros((p, total_T))
    errors = np.random.normal(loc=0.0, scale=sd, size=(p, total_T))
    X[:, :lag] = errors[:, :lag] * sigma_t[:lag].T

    for t in range(lag, total_T):
        phi = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
        X[:, t] = phi + sigma_t[t] * errors[:, t]

    return X.T[burn_in:], beta, sigma_t, GC


def lorenz(x, t, F):
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F
    return dxdt


def simulate_nonstationary_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, noise_std=2.0, mean_log_sigma=2.5, burn_in=1000,
                                     seed=0):
    """
    Generate Lorenz-96 data with time-varying noise variance.

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
    noise_std : float, default 2.0
        Standard deviation of the GP
    mean_log_sigma : float, default 2.5
        Mean of the GP
    burn_in : int, default 1000
        Burn-in period
    seed : int, default 0
        Random seed

    Returns
    -------
    tuple
        (data, GC, sigma_t)— time series array of shape (T, p), ground-truth causal graph of shape (p, p), and time-varying noise scaling factor of shape (T, p).
    """
    if seed is not None:
        np.random.seed(seed)

    total_T = T + burn_in
    t = np.linspace(0, total_T * delta_t, total_T)

    x0 = np.random.normal(scale=0.01, size=p)
    X = odeint(lorenz, x0, t, args=(F,))

    time = np.linspace(0, 1, total_T)
    K = rbf_kernel(time, l=0.2)
    K += 1e-4 * np.eye(len(K))
    sigma_t = np.zeros((total_T, p))
    log_sigma_t = np.random.multivariate_normal(
        mean=np.ones(total_T) * mean_log_sigma,
        cov=(noise_std ** 2) * K
    )
    sigma_t_shared = np.exp(log_sigma_t)
    sigma_t = np.tile(sigma_t_shared[:, None], (1, p))

    noise = np.random.normal(loc=0.0, scale=sd, size=(total_T, p)) * sigma_t
    X += noise

    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC, sigma_t
