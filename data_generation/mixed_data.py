"""
Mixed data generation for VAR and Lorenz 96 datasets.

Reference:
    [1] https://github.com/chunhuiz/MiTCD/blob/main/data_process.py
"""

import numpy as np
import torch
import random
import os
from vanilla import simulate_var, simulate_lorenz_96
from sklearn import preprocessing
from copy import deepcopy


def generate_mixed_var(p, T, lag=3, typeflag=None, discrete_ratio=0.3, seed=0, length_per_batch=50,
                       device=torch.device('cuda:1')):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if typeflag is None:
        num_discrete = int(round(p * discrete_ratio))
        discrete_indices = np.random.choice(p, size=num_discrete, replace=False)
        typeflag = [0 if i in discrete_indices else 1 for i in range(p)]

    X_np, beta, GC = simulate_var(p=p, T=T, lag=lag, seed=seed)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X_np)

    X_scaled_bin = X_scaled.copy()
    for i, flag in enumerate(typeflag):
        if flag == 0:
            X_scaled_bin[:, i] = X_scaled_bin[:, i] > 0.5

    X_pre = torch.tensor(X_scaled[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(p):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm

    X_real_tensor = torch.tensor(X_real, dtype=torch.float32, device=device)
    X_bin_inst = deepcopy(X_real_tensor)
    for i in range(p):
        if typeflag[i] == 0:
            X_bin_inst[:, :, i] = X_bin_inst[:, :, i] > 0.5

    return X_scaled_bin.astype(np.float32), X_bin_inst.cpu().numpy(), GC


def generate_mixed_lorenz_96(p, T, F=10, typeflag=None, discrete_ratio=0.3, seed=0, length_per_batch=50,
                             device=torch.device('cuda:1')):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if typeflag is None:
        num_discrete = int(round(p * discrete_ratio))
        discrete_indices = np.random.choice(p, size=num_discrete, replace=False)
        typeflag = [0 if i in discrete_indices else 1 for i in range(p)]

    X_np, GC = simulate_lorenz_96(p=p, T=T, F=F, seed=seed)

    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X_np)

    X_scaled_bin = X_scaled.copy()
    for i, flag in enumerate(typeflag):
        if flag == 0:
            X_scaled_bin[:, i] = X_scaled_bin[:, i] > 0.5

    # instance norm
    X_pre = torch.tensor(X_scaled[np.newaxis], dtype=torch.float32).reshape(-1, length_per_batch, p).numpy()
    X_real = np.zeros_like(X_pre)
    for i in range(X_real.shape[0]):
        for j in range(p):
            instance = X_pre[i, :, j]
            instance_norm = (instance - np.min(instance)) / (np.max(instance) - np.min(instance))
            X_real[i, :, j] = instance_norm

    X_real_tensor = torch.tensor(X_real, dtype=torch.float32, device=device)
    X_bin_inst = deepcopy(X_real_tensor)
    for i in range(p):
        if typeflag[i] == 0:
            X_bin_inst[:, :, i] = X_bin_inst[:, :, i] > 0.5

    return X_scaled_bin.astype(np.float32), X_bin_inst.cpu().numpy(), GC


def save_mixed_data(data_bin_global, data_bin_inst, gc, dataset_type, p, T, seed, F=None, discrete_ratio=0.3):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "mixed_data")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == 'VAR':
        filename = f'mixed_data_ratio{discrete_ratio}_VAR_p{p}_T{T}_seed{seed}.npz'
    elif dataset_type == 'Lorenz':
        filename = f'mixed_data_ratio{discrete_ratio}_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz'
    else:
        raise ValueError("Unsupported dataset_type")

    path = os.path.join(output_dir, filename)
    np.savez(path, data_bin_global=data_bin_global, data_bin_inst=data_bin_inst, gc=gc)


def load_and_check_mixed_data(ps, Ts, Fs, discrete_ratio):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "mixed_data")

    print("\n--- Checking mixed VAR data ---")
    for p in ps:
        for T in Ts:
            filename = os.path.join(output_dir, f"mixed_data_ratio{discrete_ratio}_VAR_p{p}_T{T}_seed0.npz")
            if os.path.exists(filename):
                data = np.load(filename)
                print(f"Loaded {filename}")
                print(f"data_bin_global shape: {data['data_bin_global'].shape}")
                print(f"data_bin_inst shape: {data['data_bin_inst'].shape}")
                print(f"gc shape: {data['gc'].shape}")
            else:
                print(f"{filename} not found!")

    print("\n--- Checking mixed Lorenz data ---")
    for p in ps:
        for T in Ts:
            for F in Fs:
                filename = os.path.join(output_dir, f"mixed_data_ratio{discrete_ratio}_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
                if os.path.exists(filename):
                    data = np.load(filename)
                    print(f"Loaded {filename}")
                    print(f"data_bin_global shape: {data['data_bin_global'].shape}")
                    print(f"data_bin_inst shape: {data['data_bin_inst'].shape}")
                    print(f"gc shape: {data['gc'].shape}")
                else:
                    print(f"{filename} not found!")


if __name__ == "__main__":
    ps = [10]  # [10, 15]
    Ts = [500, 1000]
    Fs = [10, 40]
    seeds = range(5)
    lag = 3
    discrete_ratio = 0.5

    for p in ps:
        for T in Ts:
            for seed in seeds:
                print(f"Generating mixed VAR: p={p}, T={T}, seed={seed}")
                data_bin_global, data_bin_inst, gc = generate_mixed_var(p=p, T=T, lag=lag,
                                                                        discrete_ratio=discrete_ratio, seed=seed)
                save_mixed_data(data_bin_global, data_bin_inst, gc, dataset_type='VAR', p=p, T=T, seed=seed,
                                discrete_ratio=discrete_ratio)

    for p in ps:
        for T in Ts:
            for F in Fs:
                for seed in seeds:
                    print(f"Generating mixed Lorenz: p={p}, T={T}, F={F}, seed={seed}")
                    data_bin_global, data_bin_inst, gc = generate_mixed_lorenz_96(p=p, T=T, F=F,
                                                                                  discrete_ratio=discrete_ratio,
                                                                                  seed=seed)
                    save_mixed_data(data_bin_global, data_bin_inst, gc, dataset_type='Lorenz', p=p, T=T, F=F, seed=seed,
                                    discrete_ratio=discrete_ratio)

    load_and_check_mixed_data(ps, Ts, Fs, discrete_ratio=discrete_ratio)
