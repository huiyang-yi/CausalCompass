"""
Non-Gaussian noise data generation for VAR models.
Uses exponential distribution instead of Gaussian noise.
"""
import numpy as np
import os
import sys
from copy import deepcopy
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics.pairwise import rbf_kernel as sklearn_rbf_kernel

# Import necessary functions from existing files
sys.path.append(os.path.dirname(__file__))


def rbf_kernel(t, l):
    """RBF kernel for nonstationary scenario"""
    t = np.array(t).reshape(-1, 1)
    return sklearn_rbf_kernel(t, gamma=1 / (2 * l ** 2))


def make_var_stationary(beta, radius=0.97):
    """Rescale coefficients of VAR model to make stable."""
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


def generate_exponential_noise(shape, scale=0.1):
    """
    Generate exponential noise centered at zero.

    Parameters
    ----------
    shape : tuple
        Shape of noise array
    scale : float
        Scale parameter (related to standard deviation)

    Returns
    -------
    noise : np.ndarray
        Exponential noise centered at zero

    Notes
    -----
    Exponential distribution has mean = scale, std = scale
    We shift it to have mean = 0 for fair comparison with Gaussian
    Random seed should be set externally before calling this function.
    """
    # Generate exponential noise: Exp(1/scale)
    # For exponential distribution: mean = scale, std = scale
    noise = np.random.exponential(scale=scale, size=shape)

    # Center the noise (subtract mean to make it zero-mean)
    noise = noise - scale

    return noise


# ==================== Vanilla ====================
def simulate_var_non_gaussian(p, T, lag, sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    """Generate VAR data with exponential noise"""
    if seed is not None:
        np.random.seed(seed)

    # Set up coefficients and Granger causality ground truth
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

    # Generate data with exponential noise
    burn_in = 100
    errors = generate_exponential_noise((p, T + burn_in), scale=sd)
    X = np.zeros((p, T + burn_in))
    X[:, :lag] = errors[:, :lag]
    for t in range(lag, T + burn_in):
        X[:, t] = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
        X[:, t] += errors[:, t - 1]

    return X.T[burn_in:], beta, GC


# ==================== Standardized ====================
def simulate_var_standardized_non_gaussian(p, T, lag=3, method='zscore', seed=0):
    """Generate standardized VAR data with exponential noise"""
    X_np, beta, GC = simulate_var_non_gaussian(p, T, lag, seed=seed)

    if method == 'zscore':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError("Normalization method must be 'zscore' or 'minmax'")

    X_scaled = scaler.fit_transform(X_np)
    return X_scaled.astype(np.float32), beta, GC


# ==================== Trend & Season ====================
def simulate_var_trendseason_non_gaussian(p, T, lag=3, sparsity=0.2, beta_value=1.0, sd=0.1,
                                          trend_strength=0.01, season_strength=0.5,
                                          season_periods=12, seed=0):
    """Generate VAR data with trend, seasonality and exponential noise"""
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

    # Use exponential noise instead of Gaussian
    errors = generate_exponential_noise((p, total_T), scale=sd)

    t_idx = np.arange(total_T)
    base_period = float(season_periods)

    # Trend component
    trend = np.zeros((p, total_T))
    for j in range(p):
        trend_modifier = (j + 1) * 0.5
        trend[j, :] = trend_strength * trend_modifier * t_idx

    # Seasonal component
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


# ==================== Confounder ====================
def simulate_var_confounder_non_gaussian(p, T, lag=3, sparsity=0.2, beta_value=1.0, sd=0.1,
                                         rho=0.5, seed=0):
    """Generate VAR data with cross-lag confounders and exponential noise"""
    if seed is not None:
        np.random.seed(seed)

    total_vars = 2 * p  # p confounders + p observed variables

    # Set up coefficients and Granger causality ground truth for observed variables
    GC = np.eye(p, dtype=int)
    beta_obs = np.eye(p) * beta_value

    num_nonzero = int(p * sparsity) - 1
    for i in range(p):
        choice = np.random.choice(p - 1, size=num_nonzero, replace=False)
        choice[choice >= i] += 1
        beta_obs[i, choice] = beta_value
        GC[i, choice] = 1

    # Create cross-lag confounder structure
    C0 = np.zeros((p, p))  # contemporaneous confounder effects
    C_lags = [np.zeros((p, p)) for _ in range(lag)]  # lagged confounder effects

    for i in range(p):  # target variable
        for j in range(p):  # source variable
            if i == j or GC[i, j] == 1:  # skip self-loops and existing edges
                continue

            if np.random.binomial(1, rho) == 1:
                k = np.random.choice(p)  # choose confounder Uk

                # Contemporaneous: Uk(t) -> Xj(t)
                C0[j, k] = beta_value

                # Cross-lag: Uk(t-τ) -> Xi(t) for all τ=1..lag
                for tau in range(lag):
                    C_lags[tau][i, k] = beta_value

    # Construct multi-lag coefficient matrix
    blocks = []
    for tau in range(lag):
        B = np.zeros((total_vars, total_vars))

        # Confounder dynamics (diagonal VAR)
        B[:p, :p] = np.eye(p) * beta_value

        # Observed variable dynamics
        B[p:, p:] = beta_obs

        # Cross-lag confounder effects
        B[p:, :p] = C_lags[tau]

        blocks.append(B)

    beta = np.hstack(blocks)
    beta = make_var_stationary(beta)

    burn_in = 100
    errors = generate_exponential_noise((total_vars, T + burn_in), scale=sd)
    X = np.zeros((total_vars, T + burn_in))
    X[:, :lag] = errors[:, :lag]

    for t in range(lag, T + burn_in):
        # VAR dynamics
        lagged = np.dot(beta, X[:, (t - lag):t].flatten(order='F')) + errors[:, t - 1]

        # Split confounders and observed variables
        U_t = lagged[:p]
        X_t = lagged[p:]

        # Add contemporaneous confounder effects
        X_t += np.dot(C0, U_t)

        # Update state
        X[:p, t] = U_t
        X[p:, t] = X_t

    # Extract observed variables and remove burn-in
    X_observed = X[p:, burn_in:].T

    return X_observed, beta, GC


# ==================== Missing Data ====================
def simulate_var_missing_non_gaussian(p, T, lag=3, seed=0, missing_prob=0.4):
    """Generate VAR data with missing values and exponential noise"""
    if seed is not None:
        np.random.seed(seed)

    X_np, beta, GC = simulate_var_non_gaussian(p=p, T=T, lag=lag, seed=seed)

    # Create mask (reset seed like original implementation)
    if seed is not None:
        np.random.seed(seed)
    mask = np.random.choice([0, 1], size=(T, p), p=[missing_prob, 1 - missing_prob])

    # Apply mask and interpolate (zero-order hold)
    sampled = X_np * mask
    interp_data = interp_masked_data_zoh(sampled, mask)

    return interp_data, sampled, mask, GC, X_np


def interp_masked_data_zoh(data, mask):
    """Zero-order hold interpolation for missing data"""
    T, p = data.shape
    new_data = deepcopy(data)
    for i in range(1, T):
        new_data[i] = data[i] * mask[i] + new_data[i - 1] * (1 - mask[i])
    return new_data


# ==================== Measurement Error ====================
def simulate_var_measurement_error_non_gaussian(p, T, lag, gamma=1.2,
                                                sparsity=0.2, beta_value=1.0, sd=0.1, seed=0):
    """Generate VAR data with measurement error and exponential noise"""
    X_clean, beta, GC = simulate_var_non_gaussian(
        p=p, T=T, lag=lag,
        sparsity=sparsity, beta_value=beta_value, sd=sd, seed=seed
    )

    # Add measurement error (use current random state, no seed reset)
    X_var = np.var(X_clean, axis=0)
    err_var = gamma * X_var
    err_std = np.sqrt(err_var)

    # Use Gaussian for measurement error (standard practice)
    measurement_noise = np.random.standard_normal(size=(T, p)) * err_std[np.newaxis, :]

    X_meas = X_clean + measurement_noise
    return X_meas, beta, GC


def simulate_var_mixed_non_gaussian(p, T, lag=3, discrete_ratio=0.5, seed=0):
    """Generate mixed continuous/discrete VAR data with exponential noise"""
    if seed is not None:
        np.random.seed(seed)

    # Determine which variables are discrete
    num_discrete = int(round(p * discrete_ratio))
    discrete_indices = np.random.choice(p, size=num_discrete, replace=False)
    typeflag = [0 if i in discrete_indices else 1 for i in range(p)]

    # Generate VAR data
    X_np, beta, GC = simulate_var_non_gaussian(p=p, T=T, lag=lag, seed=seed)

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_np)

    # Global binarization
    X_scaled_bin = X_scaled.copy()
    for i, flag in enumerate(typeflag):
        if flag == 0:  # discrete variable
            X_scaled_bin[:, i] = X_scaled_bin[:, i] > 0.5

    return X_scaled_bin.astype(np.float32), GC, typeflag


# ==================== Nonstationary ====================
def simulate_var_nonstationary_non_gaussian(p, T, lag=3, sparsity=0.2, sd=0.1,
                                            beta_value=1.0, noise_std=1.0,
                                            mean_log_sigma=1.0, seed=0):
    """Generate nonstationary VAR data with time-varying variance and exponential noise"""
    if seed is not None:
        np.random.seed(seed)

    # Build fixed Granger Causality graph
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

    # GP generate time-varying σ_i(t)
    time = np.linspace(0, 1, total_T)
    K = rbf_kernel(time, l=0.2)
    K += 1e-4 * np.eye(len(K))

    log_sigma_t = np.random.multivariate_normal(
        mean=np.ones(total_T) * mean_log_sigma,
        cov=(noise_std ** 2) * K
    )
    sigma_t_shared = np.exp(log_sigma_t)
    sigma_t = np.tile(sigma_t_shared[:, None], (1, p))

    # Generate exponential noise with time-varying scale
    X = np.zeros((p, total_T))

    # Generate all errors at once
    errors = generate_exponential_noise((p, total_T), scale=sd)

    # Initialize with scaled exponential noise
    for t in range(lag):
        X[:, t] = errors[:, t] * sigma_t[t]

    for t in range(lag, total_T):
        phi = np.dot(beta, X[:, (t - lag):t].flatten(order='F'))
        X[:, t] = phi + sigma_t[t] * errors[:, t]

    return X.T[burn_in:], beta, sigma_t, GC


# ==================== Save & Load Functions ====================
def save_non_gaussian_data(data, beta_or_gc, scenario, p, T, seed, **kwargs):
    """
    Save non-Gaussian data with scenario-specific parameters

    Parameters
    ----------
    data : np.ndarray
        Time series data
    beta_or_gc : np.ndarray
        Coefficient matrix or GC matrix
    scenario : str
        One of: vanilla, standardized, trendseason, confounder, missing,
                measurement_error, mixed_data, nonstationary
    p : int
        Number of variables
    T : int
        Number of time steps
    seed : int
        Random seed
    **kwargs : dict
        Additional scenario-specific data
    """
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    output_dir = os.path.join(parent_dir, "datasets", "non_gaussian", scenario)
    os.makedirs(output_dir, exist_ok=True)

    # Build filename based on scenario
    if scenario == "vanilla":
        filename = f'non_gaussian_vanilla_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {'data':data, 'beta':beta_or_gc, 'gc':kwargs.get('gc')}

    elif scenario == "standardized":
        method = kwargs.get('method', 'zscore')
        filename = f'non_gaussian_standardized_{method}_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {'data':data, 'gc':beta_or_gc}

    elif scenario == "trendseason":
        filename = f'non_gaussian_trendseason_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {'data':data, 'beta':beta_or_gc, 'gc':kwargs.get('gc')}

    elif scenario == "confounder":
        rho = kwargs.get('rho', 0.5)
        filename = f'non_gaussian_confounder_rho{rho}_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {'data':data, 'beta':beta_or_gc, 'gc':kwargs.get('gc')}

    elif scenario == "missing":
        missing_prob = kwargs.get('missing_prob', 0.4)
        filename = f'non_gaussian_missing_prob{missing_prob}_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {
            'data_interp':data,
            'data_masked':kwargs.get('data_masked'),
            'mask':kwargs.get('mask'),
            'gc':beta_or_gc,
            'original_data':kwargs.get('original_data')
        }

    elif scenario == "measurement_error":
        gamma = kwargs.get('gamma', 1.2)
        filename = f'non_gaussian_ME_gamma{gamma}_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {'data':data, 'beta':beta_or_gc, 'gc':kwargs.get('gc')}

    elif scenario == "mixed_data":
        ratio = kwargs.get('discrete_ratio', 0.5)
        filename = f'non_gaussian_mixed_data_ratio{ratio}_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {'data_bin_global':data, 'gc':beta_or_gc,
                     'typeflag':kwargs.get('typeflag')}

    elif scenario == "nonstationary":
        noise_std = kwargs.get('noise_std', 1.0)
        mean_log_sigma = kwargs.get('mean_log_sigma', 1.0)
        filename = f'non_gaussian_nonstationary_noisestd{noise_std}_mean{mean_log_sigma}_VAR_p{p}_T{T}_seed{seed}.npz'
        save_dict = {
            'data':data,
            'beta':beta_or_gc,
            'sigma_t':kwargs.get('sigma_t'),
            'gc':kwargs.get('gc')
        }
    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    path = os.path.join(output_dir, filename)
    np.savez(path, **save_dict)


def load_and_check_non_gaussian_data(ps, Ts, scenarios):
    """Load and verify saved non-Gaussian data"""
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    base_dir = os.path.join(parent_dir, "datasets", "non_gaussian")

    print("\n" + "=" * 70)
    print("Checking Non-Gaussian VAR Data")
    print("=" * 70)

    for scenario in scenarios:
        print(f"\n--- {scenario.upper()} ---")
        scenario_dir = os.path.join(base_dir, scenario)

        for p in ps:
            for T in Ts:
                # Different scenarios have different filename patterns
                if scenario == "vanilla":
                    pattern = f'non_gaussian_vanilla_VAR_p{p}_T{T}_seed0.npz'
                elif scenario == "standardized":
                    for method in ['zscore', 'minmax']:
                        pattern = f'non_gaussian_standardized_{method}_VAR_p{p}_T{T}_seed0.npz'
                        check_file(scenario_dir, pattern)
                    continue
                elif scenario == "trendseason":
                    pattern = f'non_gaussian_trendseason_VAR_p{p}_T{T}_seed0.npz'
                elif scenario == "confounder":
                    pattern = f'non_gaussian_confounder_rho0.5_VAR_p{p}_T{T}_seed0.npz'
                elif scenario == "missing":
                    pattern = f'non_gaussian_missing_prob0.4_VAR_p{p}_T{T}_seed0.npz'
                elif scenario == "measurement_error":
                    pattern = f'non_gaussian_ME_gamma1.2_VAR_p{p}_T{T}_seed0.npz'
                elif scenario == "mixed_data":
                    pattern = f'non_gaussian_mixed_data_ratio0.5_VAR_p{p}_T{T}_seed0.npz'
                elif scenario == "nonstationary":
                    pattern = f'non_gaussian_nonstationary_noisestd1.0_mean1.0_VAR_p{p}_T{T}_seed0.npz'
                else:
                    continue

                check_file(scenario_dir, pattern)


def check_file(directory, filename):
    """Helper function to check if file exists and print info"""
    path = os.path.join(directory, filename)
    if os.path.exists(path):
        data = np.load(path)
        print(f"✓ {filename}")
        for key in data.files:
            print(f"  - {key}: {data[key].shape}")
    else:
        print(f"✗ {filename} NOT FOUND")


# ==================== Main Generation Script ====================
if __name__ == "__main__":
    ps = [10]
    Ts = [1000]
    seeds = range(5)
    lag = 3

    print("=" * 70)
    print("Generating Non-Gaussian (Exponential Noise) VAR Data")
    print("=" * 70)
    print(f"Parameters: p={ps}, T={Ts}, seeds={list(seeds)}, lag={lag}")
    print("Noise type: Exponential (centered at zero)")
    print("=" * 70)

    # 1. Vanilla
    print("\n[1/8] Generating Vanilla...")
    for p in ps:
        for T in Ts:
            for seed in seeds:
                data, beta, gc = simulate_var_non_gaussian(p, T, lag, seed=seed)
                save_non_gaussian_data(data, beta, "vanilla", p, T, seed, gc=gc)
                print(f"  ✓ p={p}, T={T}, seed={seed}")

    # 2. Standardized
    print("\n[2/8] Generating Standardized...")
    for method in ['zscore', 'minmax']:
        for p in ps:
            for T in Ts:
                for seed in seeds:
                    data, beta, gc = simulate_var_standardized_non_gaussian(p, T, lag, method, seed)
                    save_non_gaussian_data(data, gc, "standardized", p, T, seed, method=method)
                    print(f"  ✓ {method}, p={p}, T={T}, seed={seed}")

    # 3. Trend & Season
    print("\n[3/8] Generating Trend & Season...")
    for p in ps:
        for T in Ts:
            for seed in seeds:
                data, beta, gc = simulate_var_trendseason_non_gaussian(p, T, lag, seed=seed)
                save_non_gaussian_data(data, beta, "trendseason", p, T, seed, gc=gc)
                print(f"  ✓ p={p}, T={T}, seed={seed}")

    # 4. Confounder
    print("\n[4/8] Generating Confounder...")
    rho = 0.5  # Your original parameter
    for p in ps:
        for T in Ts:
            for seed in seeds:
                data, beta, gc = simulate_var_confounder_non_gaussian(
                    p, T, lag, rho=rho, seed=seed
                )
                save_non_gaussian_data(data, beta, "confounder", p, T, seed,
                                       gc=gc, rho=rho)
                print(f"  ✓ p={p}, T={T}, seed={seed}")

    # 5. Missing Data
    print("\n[5/8] Generating Missing Data...")
    missing_prob = 0.4  # Your original parameter
    for p in ps:
        for T in Ts:
            for seed in seeds:
                data_interp, data_masked, mask, gc, original = simulate_var_missing_non_gaussian(
                    p, T, lag, seed, missing_prob
                )
                save_non_gaussian_data(data_interp, gc, "missing", p, T, seed,
                                       data_masked=data_masked, mask=mask,
                                       original_data=original, missing_prob=missing_prob)
                print(f"  ✓ p={p}, T={T}, seed={seed}")

    # 6. Measurement Error
    print("\n[6/8] Generating Measurement Error...")
    gamma = 1.2
    for p in ps:
        for T in Ts:
            for seed in seeds:
                data, beta, gc = simulate_var_measurement_error_non_gaussian(
                    p, T, lag, gamma, seed=seed
                )
                save_non_gaussian_data(data, beta, "measurement_error", p, T, seed,
                                       gc=gc, gamma=gamma)
                print(f"  ✓ p={p}, T={T}, seed={seed}")

    # 7. Mixed Data
    print("\n[7/8] Generating Mixed Data...")
    discrete_ratio = 0.5  # Your original parameter
    for p in ps:
        for T in Ts:
            for seed in seeds:
                data, gc, typeflag = simulate_var_mixed_non_gaussian(
                    p, T, lag, discrete_ratio, seed
                )
                save_non_gaussian_data(data, gc, "mixed_data", p, T, seed,
                                       discrete_ratio=discrete_ratio, typeflag=typeflag)
                print(f"  ✓ p={p}, T={T}, seed={seed}")

    # 8. Nonstationary
    print("\n[8/8] Generating Nonstationary...")
    noise_std = 1.0
    mean_log_sigma = 1.0
    for p in ps:
        for T in Ts:
            for seed in seeds:
                data, beta, sigma_t, gc = simulate_var_nonstationary_non_gaussian(
                    p, T, lag, noise_std=noise_std, mean_log_sigma=mean_log_sigma, seed=seed
                )
                save_non_gaussian_data(data, beta, "nonstationary", p, T, seed,
                                       gc=gc, sigma_t=sigma_t,
                                       noise_std=noise_std, mean_log_sigma=mean_log_sigma)
                print(f"  ✓ p={p}, T={T}, seed={seed}")

    # Verify all generated data
    scenarios = ["vanilla", "standardized", "trendseason", "confounder",
                 "missing", "measurement_error", "mixed_data", "nonstationary"]
    load_and_check_non_gaussian_data(ps, Ts, scenarios)

    print("\n" + "=" * 70)
    print("Non-Gaussian data generation complete!")
    print("=" * 70)
