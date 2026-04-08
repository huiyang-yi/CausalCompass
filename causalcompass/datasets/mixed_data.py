# Portions of this file are adapted from MiTCD
# Copyright (c) 2025 ChunhuiZhao
# https://github.com/chunhuiz/MiTCD
# Licensed under the MIT License
"""
Mixed data generation for VAR and Lorenz 96 datasets.

Reference:
    [1] https://github.com/chunhuiz/MiTCD/blob/main/data_process.py
"""

import numpy as np
import torch
import random
import os
from .vanilla import simulate_var, simulate_lorenz_96
from sklearn import preprocessing
from copy import deepcopy


def generate_mixed_var(p, T, lag=3, sparsity=0.2, beta_value=1.0, sd=0.1, burn_in=100, typeflag=None, discrete_ratio=0.5, seed=0, length_per_batch=50,
                       device=torch.device('cuda:0')):
    """
    Generate VAR data where a proportion of variables are discretized.

    References
    ----------
    https://github.com/chunhuiz/MiTCD

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
    typeflag : list or None, default None
        Manual specification of variable types (0=discrete, 1=continuous). If None, automatically generated based on discrete_ratio
    discrete_ratio : float, default 0.5
        Ratio of discrete variables
    seed : int, default 0
        Random seed
    length_per_batch : int, default 50
        Length of each batch for instance normalization
    device : torch.device, default torch.device('cuda:0')
        Device for tensor computation

    Returns
    -------
    tuple
        (data_bin_global, data_bin_inst, GC) — globally binarized time series of shape (T, p), instance-normalized and binarized time series of shape (num_batches, length_per_batch, p), and ground-truth causal graph of shape (p, p). data_bin_inst is used as input for the MiTCD algorithm, while all other algorithms use data_bin_global as input.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if typeflag is None:
        num_discrete = int(round(p * discrete_ratio))
        discrete_indices = np.random.choice(p, size=num_discrete, replace=False)
        typeflag = [0 if i in discrete_indices else 1 for i in range(p)]

    X_np, beta, GC = simulate_var(p=p, T=T, lag=lag, sparsity=sparsity, beta_value=beta_value, sd=sd, burn_in=burn_in, seed=seed)

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


def generate_mixed_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, burn_in=1000, typeflag=None, discrete_ratio=0.5, seed=0, length_per_batch=50,
                             device=torch.device('cuda:0')):
    """
    Generate Lorenz-96 data where a proportion of variables are discretized.

    References
    ----------
    https://github.com/chunhuiz/MiTCD

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
    typeflag : list or None, default None
        Manual specification of variable types (0=discrete, 1=continuous). If None, automatically generated based on discrete_ratio
    discrete_ratio : float, default 0.5
        Ratio of discrete variables
    seed : int, default 0
        Random seed
    length_per_batch : int, default 50
        Length of each batch for instance normalization
    device : torch.device, default torch.device('cuda:0')
        Device for tensor computation

    Returns
    -------
    tuple
        (data_bin_global, data_bin_inst, GC) — globally binarized time series of shape (T, p), instance-normalized and binarized time series of shape (num_batches, length_per_batch, p), and ground-truth causal graph of shape (p, p). data_bin_inst is used as input for the MiTCD algorithm, while all other algorithms use data_bin_global as input.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    if typeflag is None:
        num_discrete = int(round(p * discrete_ratio))
        discrete_indices = np.random.choice(p, size=num_discrete, replace=False)
        typeflag = [0 if i in discrete_indices else 1 for i in range(p)]

    X_np, GC = simulate_lorenz_96(p=p, T=T, F=F, delta_t=delta_t, sd=sd, burn_in=burn_in, seed=seed)

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
