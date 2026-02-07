"""
Trend_season data generation for VAR and Lorenz 96 datasets.


Reference:
    [1] https://github.com/hferdous/TimeGraph/blob/main/Codes/C1.ipynb
"""

import numpy as np
from scipy.integrate import odeint
import os
from vanilla import make_var_stationary


def simulate_var_with_trend_season(p, T, lag=3, sparsity=0.2, beta_value=1.0, sd=0.1,
                                   trend_strength=0.01, season_strength=0.5,
                                   season_periods=12, seed=0):
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

    burn_in = 100
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
    # -----------------------------------------------------

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
    # -----------------------------------------------------

    noise = np.random.normal(scale=sd, size=(total_T, p))
    X += noise + trend + season

    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


def save_trendseason_data(data, beta_or_gc, dataset_type, p, T, seed, F=None, gc=None):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "trendseason")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == 'VAR':
        filename = f'trendseason_VAR_p{p}_T{T}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, beta=beta_or_gc, gc=gc)
    elif dataset_type == 'Lorenz':
        filename = f'trendseason_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, gc=beta_or_gc)


def load_and_check_trendseason_data(ps, Ts, Fs):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "trendseason")

    print("\n--- Checking trendseason VAR data ---")
    for p in ps:
        for T in Ts:
            filename = os.path.join(output_dir, f"trendseason_VAR_p{p}_T{T}_seed0.npz")
            if os.path.exists(filename):
                data = np.load(filename)
                print(f"Loaded {filename}")
                print(
                    f"data shape: {data['data'].shape}, beta shape: {data['beta'].shape}, gc shape: {data['gc'].shape}")
            else:
                print(f"{filename} not found!")

    print("\n--- Checking trendseason Lorenz data ---")
    for p in ps:
        for T in Ts:
            for F in Fs:
                filename = os.path.join(output_dir, f"trendseason_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
                if os.path.exists(filename):
                    data = np.load(filename)
                    print(f"Loaded {filename}")
                    print(f"data shape: {data['data'].shape}, gc shape: {data['gc'].shape}")
                else:
                    print(f"{filename} not found!")


if __name__ == "__main__":
    ps = [10]  # [10, 15]
    Ts = [500, 1000]
    Fs = [10, 40]
    seeds = range(5)
    lag = 3

    for p in ps:
        for T in Ts:
            for seed in seeds:
                print(f"Generating trendseason VAR: p={p}, T={T}, seed={seed}")
                data, beta, gc = simulate_var_with_trend_season(p, T, lag=lag, trend_strength=0.01, season_strength=0.5,
                                                                season_periods=12, seed=seed)
                save_trendseason_data(data, beta, dataset_type='VAR', p=p, T=T, seed=seed, gc=gc)

    for p in ps:
        for T in Ts:
            for F in Fs:
                for seed in seeds:
                    print(f"Generating trendseason Lorenz: p={p}, T={T}, F={F}, seed={seed}")
                    data, gc = simulate_lorenz_with_trend_season(p, T, F=F, trend_strength=0.01, season_strength=0.5,
                                                                 season_periods=12, seed=seed)
                    save_trendseason_data(data, gc, dataset_type='Lorenz', p=p, T=T, F=F, seed=seed)

    load_and_check_trendseason_data(ps, Ts, Fs)
