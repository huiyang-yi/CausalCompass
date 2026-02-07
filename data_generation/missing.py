"""
Missing data generation for VAR and Lorenz 96 datasets.
This script creates datasets with various missing data patterns
and saves them in .npz format.

Reference:
    [1] https://github.com/jarrycyx/UNN/blob/main/CUTS_Plus/cuts_plus_example.ipynb
"""

import numpy as np
import os
from copy import deepcopy
from einops import rearrange
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor
from vanilla import simulate_var, simulate_lorenz_96


def generate_missing_var(p, T, lag=3, seed=0, missing_config=None, interp="zoh"):
    if seed is not None:
        np.random.seed(seed)

    X_np, beta, GC = simulate_var(p=p, T=T, lag=lag, seed=seed)
    mask = create_mask(T, p, missing_config, seed=seed)
    sampled = X_np * mask
    interp_data = interp_multivar_data(sampled, mask, interp=interp)
    return interp_data, sampled, mask, GC, X_np


def generate_missing_lorenz_96(p, T, F=10, seed=0, missing_config=None, interp="zoh"):
    if seed is not None:
        np.random.seed(seed)

    X_np, GC = simulate_lorenz_96(p=p, T=T, F=F, seed=seed)
    mask = create_mask(T, p, missing_config, seed=seed)

    sampled = X_np * mask
    interp_data = interp_multivar_data(sampled, mask, interp=interp)
    return interp_data, sampled, mask, GC, X_np


def create_mask(T, N, config, seed=None):
    if seed is not None:
        np.random.seed(seed)

    mask = np.ones((T, N), dtype=np.float32)
    if config is None:
        return mask

    if "period" in config:
        period = config["period"]
        assert len(period) == N
        mask *= 0
        for i in range(N):
            mask[::period[i], i] = 1

    elif "random_period" in config:
        choices = config["random_period"]["choices"]
        prob = config["random_period"]["prob"]

        period = np.random.choice(choices, N, p=prob)
        mask *= 0
        for i in range(N):
            mask[::period[i], i] = 1

    elif "random_missing" in config:
        p = config["random_missing"]["missing_prob"]
        varlist = config["random_missing"]["missing_var"]

        if varlist == "all":
            mask = np.random.choice([0, 1], size=(T, N), p=[p, 1 - p])
        else:
            for i in varlist:
                mask[:, i] = np.random.choice([0, 1], size=(T,), p=[p, 1 - p])
    else:
        raise ValueError("Unrecognized missing config.")
    return mask


def save_missing_data(data_interp, data_masked, mask, gc, dataset_type, p, T, seed, F=None, original_data=None,
                      missing_prob=0.2):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    out_dir = os.path.join(parent_dir, "datasets", "missing")
    os.makedirs(out_dir, exist_ok=True)

    if dataset_type == "VAR":
        filename = f'missing_prob{missing_prob}_VAR_p{p}_T{T}_seed{seed}.npz'
    else:
        filename = f'missing_prob{missing_prob}_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz'

    save_dict = {
        'data_interp':data_interp.squeeze(-1).astype(np.float32),
        'data_masked':data_masked.astype(np.float32),
        'mask':mask.astype(np.float32),
        'gc':gc
    }

    if original_data is not None:
        save_dict['original_data'] = original_data.astype(np.float32)

    np.savez(os.path.join(out_dir, filename), **save_dict)


def load_and_check_missing_data(ps, Ts, Fs, missing_prob=0.2):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "missing")

    print("\n--- Checking missing VAR data ---")
    for p in ps:
        for T in Ts:
            path = os.path.join(output_dir, f"missing_prob{missing_prob}_VAR_p{p}_T{T}_seed0.npz")
            if os.path.exists(path):
                data = np.load(path)
                print(f"Loaded {path}")
                print(f"original_data shape: {data['original_data'].shape}")
                print(f"data_interp shape: {data['data_interp'].shape}")
                print(f"data_masked shape: {data['data_masked'].shape}")
                print(f"mask shape: {data['mask'].shape}")
                print(f"gc shape: {data['gc'].shape}")
            else:
                print(f"{path} not found!")

    print("\n--- Checking missing Lorenz data ---")
    for p in ps:
        for T in Ts:
            for F in Fs:
                path = os.path.join(output_dir, f"missing_prob{missing_prob}_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
                if os.path.exists(path):
                    data = np.load(path)
                    print(f"Loaded {path}")
                    print(f"original_data shape: {data['original_data'].shape}")
                    print(f"data_interp shape: {data['data_interp'].shape}")
                    print(f"data_masked shape: {data['data_masked'].shape}")
                    print(f"mask shape: {data['mask'].shape}")
                    print(f"gc shape: {data['gc'].shape}")
                else:
                    print(f"{path} not found!")


def interp_masked_data(data, mask, interp="zoh"):
    if interp == "zoh":
        T, D = data.shape
        new_data = deepcopy(data)
        for i in range(1, T):
            new_data[i] = data[i] * mask[i] + new_data[i - 1] * (1 - mask[i])
        return new_data
    else:
        x = np.argwhere(mask > 0)[:, 0]
        func = interpolate.interp1d(x, data[x, :], kind=interp,
                                    axis=0, copy=True, fill_value="extrapolate")
        new_x = np.arange(0, data.shape[0])
        return func(new_x)


def interp_multivar_data(data, mask, interp="zoh"):
    if data.ndim == 2:
        data = data[..., np.newaxis]  # (T, p, 1)
        mask = mask[..., np.newaxis]  # (T, p, 1)

    if interp == "GP":
        return interp_multivar_with_gauss_process(data, mask)
    else:
        new_data = np.zeros_like(data)
        T, N, D = data.shape
        for node_i in range(N):
            new_data[:, node_i, :] = interp_masked_data(data[:, node_i, :], mask[:, node_i], interp=interp)
        return new_data


def interp_multivar_with_gauss_process(data, mask):
    if data.ndim == 2:
        data = data[..., np.newaxis]  # (T, p, 1)
        mask = mask[..., np.newaxis]  # (T, p, 1)

    data = data * mask
    x = rearrange(data[:-1], "t n d -> t (n d)")
    y = rearrange(data[1:], "t n d -> t (n d)")

    gpr = GaussianProcessRegressor(random_state=0).fit(x, y)
    pred = gpr.predict(y)
    pred = rearrange(pred, "t (n d) -> t n d", n=data.shape[1])

    new_data = np.zeros_like(data)
    new_data[1:] = (mask * data)[1:] + (1 - mask)[1:] * pred
    new_data[0] = data[0]

    return new_data


if __name__ == "__main__":
    ps = [10]  # [10, 15]
    Ts = [500, 1000]
    Fs = [10, 40]
    seeds = range(5)
    missing_prob = 0.4

    config = {
        "random_missing":{
            "missing_prob":0.4,
            "missing_var":"all",
        }
    }

    for p in ps:
        for T in Ts:
            for seed in seeds:
                print(f"Generating missing VAR: p={p}, T={T}, seed={seed}")
                data_interp, data_masked, mask, gc, original_data = generate_missing_var(p, T, lag=3, seed=seed,
                                                                                         missing_config=config)
                save_missing_data(data_interp, data_masked, mask, gc, dataset_type='VAR', p=p, T=T, seed=seed,
                                  original_data=original_data, missing_prob=missing_prob)

    for p in ps:
        for T in Ts:
            for F in Fs:
                for seed in seeds:
                    print(f"Generating missing Lorenz: p={p}, T={T}, F={F}, seed={seed}")
                    data_interp, data_masked, mask, gc, original_data = generate_missing_lorenz_96(p, T, F=F,
                                                                                                   seed=seed,
                                                                                                   missing_config=config)
                    save_missing_data(data_interp, data_masked, mask, gc, dataset_type='Lorenz', p=p, T=T, F=F,
                                      seed=seed, original_data=original_data, missing_prob=missing_prob)

    load_and_check_missing_data(ps, Ts, Fs, missing_prob=missing_prob)
