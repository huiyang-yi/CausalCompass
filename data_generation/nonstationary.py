"""
Nonstationary data generation for VAR and Lorenz-96 systems.

This module supports:
1. Time-varying noise variance (all methods)
2. Time-varying coefficients (VAR only)

Both variations use Gaussian Process to generate smooth temporal changes.
"""

import numpy as np
from scipy.integrate import odeint
from vanilla import make_var_stationary
import os
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel


def rbf_kernel(t, l):
    t = np.array(t).reshape(-1, 1)
    return sklearn_rbf_kernel(t, gamma=1 / (2 * l ** 2))


def simulate_nonstationary_var_timevarying_coef(p, T, lag=3, sparsity=0.2, sd=0.1,
                                                beta_value=1.0, noise_std=0.8,
                                                mean_log_sigma=0.0,
                                                coef_noise_std=0.3,
                                                seed=0):
    """

    Parameters:
    -----------
    coef_noise_std : float

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

    burn_in = 100
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
                               beta_value=1.0, noise_std=0.8, mean_log_sigma=0.0, seed=0):
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

    burn_in = 100
    total_T = T + burn_in

    # GP generate time-varying σ_i(t) with adjustable mean
    time = np.linspace(0, 1, total_T)
    K = rbf_kernel(time, l=0.2)
    K += 1e-4 * np.eye(len(K))  # Numerical stability
    sigma_t = np.zeros((total_T, p))
    # All variables share one σ(t) with adjustable mean
    log_sigma_t = np.random.multivariate_normal(
        mean=np.ones(total_T) * mean_log_sigma,  # Changed: non-zero mean
        cov=(noise_std ** 2) * K
    )
    sigma_t_shared = np.exp(log_sigma_t)
    # Apply to all variables (maintain equal variance across variables)
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


def simulate_nonstationary_lorenz_96(p, T, F=10.0, delta_t=0.1, sd=0.1, noise_std=0.8, mean_log_sigma=0.0, burn_in=1000,
                                     seed=0):
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
    # All variables share one σ(t) with adjustable mean
    log_sigma_t = np.random.multivariate_normal(
        mean=np.ones(total_T) * mean_log_sigma,  # Changed: non-zero mean
        cov=(noise_std ** 2) * K
    )
    sigma_t_shared = np.exp(log_sigma_t)
    # Apply to all variables (maintain equal variance across variables)
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


def save_nonstationary_data(data, beta_or_gc, sigma_t, dataset_type, p, T, seed,
                            F=None, gc=None, noise_std=0.8, mean_log_sigma=0.0,
                            coef_noise_std=None, beta_base=None, is_timevarying_coef=False):
    """

    ---------
    coef_noise_std : float or None
    beta_base : array or None
    is_timevarying_coef : bool

    """
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "nonstationary")
    os.makedirs(output_dir, exist_ok=True)

    if dataset_type == 'VAR':
        if is_timevarying_coef:
            filename = f'nonstationary_coefstd{coef_noise_std}_noisestd{noise_std}_mean{mean_log_sigma}_VAR_p{p}_T{T}_seed{seed}.npz'
            np.savez(os.path.join(output_dir, filename),
                     data=data,
                     beta_t=beta_or_gc,
                     beta_base=beta_base,
                     sigma_t=sigma_t,
                     gc=gc,
                     coef_noise_std=coef_noise_std)
        else:
            filename = f'nonstationary_noisestd{noise_std}_mean{mean_log_sigma}_VAR_p{p}_T{T}_seed{seed}.npz'
            np.savez(os.path.join(output_dir, filename),
                     data=data,
                     beta=beta_or_gc,
                     sigma_t=sigma_t,
                     gc=gc)
    elif dataset_type == 'Lorenz':
        filename = f'nonstationary_noisestd{noise_std}_mean{mean_log_sigma}_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz'
        np.savez(os.path.join(output_dir, filename),
                 data=data,
                 sigma_t=sigma_t,
                 gc=beta_or_gc)


def check_var_timevarying_coef_data(ps, Ts, noise_std, mean_log_sigma, coef_noise_std):
    """Check VAR time-varying coefficient nonstationary data"""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "nonstationary")

    print(f"\n--- Checking VAR time-varying coef data (coef_std={coef_noise_std}, "
          f"noise_std={noise_std}, mean_log_sigma={mean_log_sigma}) ---")
    for p in ps:
        for T in Ts:
            path = os.path.join(output_dir,
                                f"nonstationary_coefstd{coef_noise_std}_noisestd{noise_std}_mean{mean_log_sigma}_VAR_p{p}_T{T}_seed0.npz")

            if os.path.exists(path):
                data = np.load(path)
                print(f"Loaded {path}")
                print(f"  data shape: {data['data'].shape}, beta_t shape: {data['beta_t'].shape}, "
                      f"beta_base shape: {data['beta_base'].shape}, "
                      f"sigma shape: {data['sigma_t'].shape}, gc shape: {data['gc'].shape}")
                print(f"  coef_noise_std: {data['coef_noise_std']}")
            else:
                print(f"{path} not found!")


def check_var_data(ps, Ts, noise_std, mean_log_sigma):
    """Check VAR nonstationary data only"""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "nonstationary")

    print(f"\n--- Checking VAR data (noise_std={noise_std}, mean_log_sigma={mean_log_sigma}) ---")
    for p in ps:
        for T in Ts:
            path = os.path.join(output_dir,
                                f"nonstationary_noisestd{noise_std}_mean{mean_log_sigma}_VAR_p{p}_T{T}_seed0.npz")
            if os.path.exists(path):
                data = np.load(path)
                print(f"Loaded {path}")
                print(f"  data shape: {data['data'].shape}, beta shape: {data['beta'].shape}, "
                      f"sigma shape: {data['sigma_t'].shape}, gc shape: {data['gc'].shape}")
            else:
                print(f"{path} not found!")


def check_lorenz_data(ps, Ts, F, noise_std, mean_log_sigma):
    """Check Lorenz nonstationary data for a specific F value"""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "nonstationary")

    print(f"\n--- Checking Lorenz data (F={F}, noise_std={noise_std}, mean_log_sigma={mean_log_sigma}) ---")
    for p in ps:
        for T in Ts:
            path = os.path.join(output_dir,
                                f"nonstationary_noisestd{noise_std}_mean{mean_log_sigma}_Lorenz_p{p}_T{T}_F{F}_seed0.npz")
            if os.path.exists(path):
                data = np.load(path)
                print(f"Loaded {path}")
                print(f"  data shape: {data['data'].shape}, sigma shape: {data['sigma_t'].shape}, "
                      f"gc shape: {data['gc'].shape}")
            else:
                print(f"{path} not found!")


if __name__ == "__main__":
    ps = [10]  # [10, 15]
    Ts = [500, 1000]
    Fs = [10, 40]
    seeds = range(5)
    lag = 3

    # var_config = {
    #     'noise_std':1.0,
    #     'mean_log_sigma':1.0
    # }

    var_timevarying_coef_config = {
        'noise_std':1.5,
        'mean_log_sigma':1.8,
        'coef_noise_std':1.0
    }

    lorenz_config = {
        10:{'noise_std':2.0, 'mean_log_sigma':2.5},
        40:{'noise_std':2.0, 'mean_log_sigma':3.5}
    }

    # print("\n" + "="*80)
    # print("Generating NOISE-ONLY time-varying VAR data")
    # print("="*80)
    # # Generate VAR data
    # for p in ps:
    #     for T in Ts:
    #         for seed in seeds:
    #             print(f"Generating nonstationary VAR: p={p}, T={T}, "
    #                   f"noise_std={var_config['noise_std']}, mean_log_sigma={var_config['mean_log_sigma']}, seed={seed}")
    #             data, beta_t, sigma_t, gc = simulate_nonstationary_var(
    #                 p, T, lag=lag,
    #                 noise_std=var_config['noise_std'],
    #                 mean_log_sigma=var_config['mean_log_sigma'],
    #                 seed=seed
    #             )
    #
    #             save_nonstationary_data(
    #                 data, beta_t, sigma_t, 'VAR', p, T, seed, gc=gc,
    #                 noise_std=var_config['noise_std'],
    #                 mean_log_sigma=var_config['mean_log_sigma'],
    #                 is_timevarying_coef=False
    #             )

    print("\n" + "=" * 80)
    print("Generating COEFFICIENT+NOISE time-varying VAR data")
    print("=" * 80)
    for p in ps:
        for T in Ts:
            for seed in seeds:
                config = var_timevarying_coef_config
                print(f"Generating time-varying coef VAR: p={p}, T={T}, "
                      f"coef_noise_std={config['coef_noise_std']}, "
                      f"noise_std={config['noise_std']}, "
                      f"mean_log_sigma={config['mean_log_sigma']}, seed={seed}")
                data, beta_t, sigma_t, gc, beta_base = simulate_nonstationary_var_timevarying_coef(
                    p, T, lag=lag,
                    noise_std=config['noise_std'],
                    mean_log_sigma=config['mean_log_sigma'],
                    coef_noise_std=config['coef_noise_std'],
                    seed=seed
                )
                save_nonstationary_data(
                    data, beta_t, sigma_t, 'VAR', p, T, seed, gc=gc,
                    noise_std=config['noise_std'],
                    mean_log_sigma=config['mean_log_sigma'],
                    coef_noise_std=config['coef_noise_std'],
                    beta_base=beta_base,
                    is_timevarying_coef=True
                )

    # # Generate Lorenz data (when uncommented)
    # for p in ps:
    #     for T in Ts:
    #         for F in Fs:
    #             config = lorenz_config[F]
    #             for seed in seeds:
    #                 print(f"Generating nonstationary Lorenz: p={p}, T={T}, F={F}, "
    #                       f"noise_std={config['noise_std']}, mean_log_sigma={config['mean_log_sigma']}, seed={seed}")
    #                 data, gc, sigma_t = simulate_nonstationary_lorenz_96(
    #                     p, T, F=F,
    #                     noise_std=config['noise_std'],
    #                     mean_log_sigma=config['mean_log_sigma'],
    #                     seed=seed
    #                 )
    #                 save_nonstationary_data(
    #                     data, gc, sigma_t, 'Lorenz', p, T, seed, F=F,
    #                     noise_std=config['noise_std'],
    #                     mean_log_sigma=config['mean_log_sigma']
    #                 )

    # # Check generated data
    # check_var_data(ps, Ts, var_config['noise_std'], var_config['mean_log_sigma'])

    check_var_timevarying_coef_data(
        ps, Ts,
        var_timevarying_coef_config['noise_std'],
        var_timevarying_coef_config['mean_log_sigma'],
        var_timevarying_coef_config['coef_noise_std']
    )

    for F in Fs:
        check_lorenz_data(ps, Ts, F, lorenz_config[F]['noise_std'], lorenz_config[F]['mean_log_sigma'])
