"""
Trend_season data generation for VAR and Lorenz 96 datasets.

Reference:
    [1] https://github.com/hferdous/TimeGraph/blob/main/Codes/C1.ipynb
"""

import numpy as np
from scipy.integrate import odeint
import os
from .vanilla import make_var_stationary


def simulate_var_with_trend_season(p, T, lag=3, sparsity=0.2, beta_value=1.0, sd=0.1,
                                   trend_strength=0.01, season_strength=0.5,
                                   season_periods=12, burn_in=100, seed=0):
    """
    Generate VAR data with additive trend and seasonal components.

    References
    ----------
    https://github.com/hferdous/TimeGraph

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
    trend_strength : float, default 0.01
        Strength of the trend
    season_strength : float, default 0.5
        Amplitude of the seasonal component
    season_periods : int, default 12
        Seasonal period
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

    total_T = T + burn_in

    errors = np.random.normal(scale=sd, size=(p, total_T))

    t_idx = np.arange(total_T)  # [0, 1, 2, ..., total_T-1]
    base_period = float(season_periods)

    trend = np.zeros((p, total_T))
    for j in range(p):
        trend_modifier = (j + 1) * 0.5
        trend[j, :] = trend_strength * trend_modifier * t_idx

    season = np.zeros((p, total_T))
    for j in range(p):
        phase_shift = 2 * np.pi * j / float(p)
        season1 = np.sin(2 * np.pi * t_idx / base_period + phase_shift)
        season2 = 0.5 * np.cos(4 * np.pi * t_idx / base_period + phase_shift)
        season[j, :] = season_strength * (season1 + season2)

    seasonal_trend = trend + season

    X = np.zeros((p, total_T))
    X[:, :lag] = errors[:, :lag] + seasonal_trend[:, :lag]

    for t in range(lag, total_T):
        phi = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
        X[:, t] = phi + errors[:, t - 1] + seasonal_trend[:, t]

    return X.T[burn_in:], beta, GC


def lorenz(x, t, F):
    p = len(x)
    dxdt = np.zeros(p)
    for i in range(p):
        dxdt[i] = (x[(i + 1) % p] - x[(i - 2) % p]) * x[(i - 1) % p] - x[i] + F
    return dxdt


def simulate_lorenz_with_trend_season(p, T, F=10.0, delta_t=0.1, sd=0.1,
                                      trend_strength=0.01, season_strength=0.5,
                                      season_periods=12, burn_in=1000, seed=0):
    """
    Generate Lorenz-96 data with additive trend and seasonal components.

    References
    ----------
    https://github.com/hferdous/TimeGraph

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
    trend_strength : float, default 0.01
        Strength of the trend
    season_strength : float, default 0.5
        Amplitude of the seasonal component
    season_periods : int, default 12
        Seasonal period
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

    total_T = T + burn_in
    t = np.linspace(0, total_T * delta_t, total_T)

    x0 = np.random.normal(scale=0.01, size=p)
    X = odeint(lorenz, x0, t, args=(F,))

    t_idx = np.arange(total_T)
    base_period = float(season_periods)

    trend = np.zeros((total_T, p))
    for j in range(p):
        trend_modifier = (j + 1) * 0.5
        trend[:, j] = trend_strength * trend_modifier * t_idx

    season = np.zeros((total_T, p))
    for j in range(p):
        phase_shift = 2 * np.pi * j / float(p)
        season1 = np.sin(2 * np.pi * t_idx / base_period + phase_shift)
        season2 = 0.5 * np.cos(4 * np.pi * t_idx / base_period + phase_shift)
        season[:, j] = season_strength * (season1 + season2)

    noise = np.random.normal(scale=sd, size=(total_T, p))
    X += noise + trend + season

    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC
