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


def simulate_var(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth.
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
    burn_in = 100
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
    if seed is not None:
        np.random.seed(seed)

    # Use scipy to solve ODE.
    x0 = np.random.normal(scale=0.01, size=p)
    t = np.linspace(0, (T + burn_in) * delta_t, T + burn_in)
    X = odeint(lorenz, x0, t, args=(F,))
    X += np.random.normal(scale=sd, size=(T + burn_in, p))

    # Set up Granger causality ground truth.
    GC = np.zeros((p, p), dtype=int)
    for i in range(p):
        GC[i, i] = 1
        GC[i, (i + 1) % p] = 1
        GC[i, (i - 1) % p] = 1
        GC[i, (i - 2) % p] = 1

    return X[burn_in:], GC


def save_data(data, beta_or_gc, dataset_type, p, T, seed, F=None, gc=None):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "vanilla")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == 'VAR':
        filename = f'vanilla_VAR_p{p}_T{T}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, beta=beta_or_gc, gc=gc)
    elif dataset_type == 'Lorenz':
        filename = f'vanilla_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz'
        path = os.path.join(output_dir, filename)
        np.savez(path, data=data, gc=beta_or_gc)


def load_and_check_data(ps, Ts, Fs):
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "vanilla")

    print("\n--- Checking saved VAR data ---")
    for p in ps:
        for T in Ts:
            filename = os.path.join(output_dir, f"vanilla_VAR_p{p}_T{T}_seed0.npz")
            if os.path.exists(filename):
                data = np.load(filename)
                print(f"Loaded {filename}")
                print(
                    f"data shape: {data['data'].shape}, beta shape: {data['beta'].shape}, gc shape: {data['gc'].shape}")
            else:
                print(f"{filename} not found!")

    print("\n--- Checking saved Lorenz data ---")
    for p in ps:
        for T in Ts:
            for F in Fs:
                filename = os.path.join(output_dir, f"vanilla_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
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
                print(f"Generating VAR data: p={p}, T={T}, seed={seed}")
                data, beta, gc = simulate_var(p=p, T=T, lag=lag, seed=seed)
                save_data(data, beta, dataset_type='VAR', p=p, T=T, seed=seed, gc=gc)

    for p in ps:
        for T in Ts:
            for F in Fs:
                for seed in seeds:
                    print(f"Generating Lorenz data: p={p}, T={T}, F={F}, seed={seed}")
                    data, gc = simulate_lorenz_96(p=p, T=T, F=F, seed=seed)
                    save_data(data, gc, dataset_type='Lorenz', p=p, T=T, F=F, seed=seed)

    load_and_check_data(ps, Ts, Fs)
