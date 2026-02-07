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
from vanilla import simulate_var, simulate_lorenz_96


def normalize_data(X_np, method):
    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'")
    return scaler.fit_transform(X_np)


def generate_standardized_var(p, T, lag=3, method='zscore', seed=0):
    X_np, beta, GC = simulate_var(p, T, lag, seed=seed)
    X_scaled = normalize_data(X_np, method)
    return X_scaled.astype(np.float32), beta, GC


def generate_standardized_lorenz_96(p, T, F=10, method='zscore', seed=0):
    X_np, GC = simulate_lorenz_96(p, T, F=F, seed=seed)
    X_scaled = normalize_data(X_np, method)
    return X_scaled.astype(np.float32), GC


def save_standardized_data(data, gc, dataset_type, method, p, T, seed, F=None):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(parent_dir, "datasets", "standardized")
    os.makedirs(out_dir, exist_ok=True)

    if dataset_type == 'VAR':
        filename = f"standardized_{method}_VAR_p{p}_T{T}_seed{seed}.npz"
    else:
        filename = f"standardized_{method}_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz"

    path = os.path.join(out_dir, filename)
    np.savez(path, data=data, gc=gc)


def load_and_check_standardized_data(ps, Ts, Fs, methods):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(parent_dir, "datasets", "standardized")

    print("\n--- Checking standardized VAR ---")
    for p in ps:
        for T in Ts:
            for method in methods:
                path = os.path.join(out_dir, f"standardized_{method}_VAR_p{p}_T{T}_seed0.npz")
                if os.path.exists(path):
                    data = np.load(path)
                    print(f"Loaded {path} | shape: {data['data'].shape}, gc: {data['gc'].shape}")
                else:
                    print(f"{path} not found!")

    print("\n--- Checking standardized Lorenz ---")
    for p in ps:
        for T in Ts:
            for F in Fs:
                for method in methods:
                    path = os.path.join(out_dir, f"standardized_{method}_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
                    if os.path.exists(path):
                        data = np.load(path)
                        print(f"Loaded {path} | shape: {data['data'].shape}, gc: {data['gc'].shape}")
                    else:
                        print(f"{path} not found!")


if __name__ == "__main__":
    ps = [10]  # [10, 15]
    Ts = [500, 1000]
    Fs = [10, 40]
    seeds = range(5)
    methods = ['zscore', 'minmax']

    for method in methods:
        for p in ps:
            for T in Ts:
                for seed in seeds:
                    print(f"Generating VAR: p={p}, T={T}, method={method}, seed={seed}")
                    data, _, gc = generate_standardized_var(p, T, method=method, seed=seed)
                    save_standardized_data(data, gc, 'VAR', method, p, T, seed)

        for p in ps:
            for T in Ts:
                for F in Fs:
                    for seed in seeds:
                        print(f"Generating Lorenz: p={p}, T={T}, F={F}, method={method}, seed={seed}")
                        data, gc = generate_standardized_lorenz_96(p, T, F=F, method=method, seed=seed)
                        save_standardized_data(data, gc, 'Lorenz', method, p, T, seed, F)

    load_and_check_standardized_data(ps, Ts, Fs, methods)
