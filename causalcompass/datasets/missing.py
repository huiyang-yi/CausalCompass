"""
Missing data generation for VAR and Lorenz 96 datasets.
This script creates datasets with various missing data patterns
and saves them in .npz format.

Reference:
    [1] https://github.com/jarrycyx/UNN/blob/main/CUTS_Plus/cuts_plus_example.ipynb
"""

import os
from copy import deepcopy

import numpy as np
from einops import rearrange
from scipy import interpolate
from sklearn.gaussian_process import GaussianProcessRegressor

from .vanilla import simulate_lorenz_96, simulate_var

SUPPORTED_MISSING_IMPUTATIONS = ("zoh", "linear", "GP")
_IMPUTATION_FILENAME_TOKENS = {
    "zoh": "zoh",
    "linear": "linear",
    "GP": "gp",
}


def normalize_missing_imputation_method(interp):
    """Return the canonical missing-data imputation method name."""
    if interp is None:
        return "zoh"

    value = str(interp).strip()
    value_lower = value.lower()
    if value_lower == "zoh":
        return "zoh"
    if value_lower == "linear":
        return "linear"
    if value_lower == "gp":
        return "GP"

    raise ValueError(
        f"Unsupported missing-data imputation method: {interp}. "
        f"Supported methods: {', '.join(SUPPORTED_MISSING_IMPUTATIONS)}"
    )



def get_missing_imputation_filename_token(interp):
    canonical = normalize_missing_imputation_method(interp)
    return _IMPUTATION_FILENAME_TOKENS[canonical]



def build_missing_dataset_filename(data_model, p, T, seed, F=None, missing_prob=0.2, interp="zoh"):
    """Build the dataset filename for a missing-data variant.

    Notes
    -----
    The historic ZOH naming is preserved for backward compatibility. Non-ZOH
    datasets add an explicit imputation marker to avoid collisions.
    """
    canonical = normalize_missing_imputation_method(interp)
    imputation_segment = ""
    if canonical != "zoh":
        imputation_segment = f"_impute{get_missing_imputation_filename_token(canonical)}"

    if data_model == "VAR":
        return f"missing_prob{missing_prob}{imputation_segment}_VAR_p{p}_T{T}_seed{seed}.npz"

    return f"missing_prob{missing_prob}{imputation_segment}_Lorenz_p{p}_T{T}_F{F}_seed{seed}.npz"



def generate_missing_var(
    p,
    T,
    lag=3,
    sparsity=0.2,
    beta_value=1.0,
    sd=0.1,
    burn_in=100,
    seed=0,
    missing_config=None,
    interp="zoh",
):
    """
    Generate VAR data with missing values and interpolation.

    References
    ----------
    https://github.com/jarrycyx/UNN

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
    seed : int, default 0
        Random seed
    missing_config : dict, default None
        Configuration for the missing pattern.

        Example
        -------
        ``missing_config = {"random_missing": {"missing_prob": 0.2,
        "missing_var": "all"}}`` means each entry is missing with probability
        ``0.2``. The ``random_missing`` pattern represents completely random
        missingness. Here, ``missing_prob`` is the missing probability and
        ``missing_var`` specifies which variables are allowed to be masked.
        Setting ``missing_var`` to ``"all"`` applies random missingness to all
        variables; alternatively, you can pass a list of variable indices such
        as ``[0, 2, 4]`` to mask only selected variables.
    interp : str, default 'zoh'
        Interpolation method. Supported values are 'zoh', 'linear', and 'GP'.

    Returns
    -------
    tuple
        (data_interp, data_masked, mask, GC, original_data) — interpolated time
        series of shape (T, p, 1), masked time series of shape (T, p), missing
        data mask of shape (T, p) where 1 indicates observed and 0 indicates
        missing, ground-truth causal graph of shape (p, p), and original
        complete time series of shape (T, p).
    """
    interp = normalize_missing_imputation_method(interp)
    if seed is not None:
        np.random.seed(seed)

    X_np, beta, GC = simulate_var(
        p=p,
        T=T,
        lag=lag,
        sparsity=sparsity,
        beta_value=beta_value,
        sd=sd,
        burn_in=burn_in,
        seed=seed,
    )
    mask = create_mask(T, p, missing_config, seed=seed)
    sampled = X_np * mask
    interp_data = interp_multivar_data(sampled, mask, interp=interp)
    return interp_data, sampled, mask, GC, X_np



def generate_missing_lorenz_96(
    p,
    T,
    F=10.0,
    delta_t=0.1,
    sd=0.1,
    burn_in=1000,
    seed=0,
    missing_config=None,
    interp="zoh",
):
    """
    Generate Lorenz-96 data with missing values and interpolation.

    Parameters
    ----------
    p : int
        Number of variables
    T : int
        Number of time points
    F : float, default 10.0
        Forcing parameter
    delta_t : float, default 0.1
        Time step for the Lorenz-96 simulator
    sd : float, default 0.1
        Noise standard deviation
    burn_in : int, default 1000
        Burn-in period
    seed : int, default 0
        Random seed
    missing_config : dict, default None
        Configuration for the missing pattern.

        Example
        -------
        ``missing_config = {"random_missing": {"missing_prob": 0.2,
        "missing_var": "all"}}`` means each entry is missing with probability
        ``0.2``. The ``random_missing`` pattern represents completely random
        missingness. Here, ``missing_prob`` is the missing probability and
        ``missing_var`` specifies which variables are allowed to be masked.
        Setting ``missing_var`` to ``"all"`` applies random missingness to all
        variables; alternatively, you can pass a list of variable indices such
        as ``[0, 2, 4]`` to mask only selected variables.
    interp : str, default 'zoh'
        Interpolation method. Supported values are 'zoh', 'linear', and 'GP'.

    Returns
    -------
    tuple
        (data_interp, data_masked, mask, GC, original_data) — interpolated time
        series of shape (T, p, 1), masked time series of shape (T, p), missing
        data mask of shape (T, p) where 1 indicates observed and 0 indicates
        missing, ground-truth causal graph of shape (p, p), and original
        complete time series of shape (T, p).
    """
    interp = normalize_missing_imputation_method(interp)
    if seed is not None:
        np.random.seed(seed)

    X_np, GC = simulate_lorenz_96(
        p=p,
        T=T,
        F=F,
        delta_t=delta_t,
        sd=sd,
        burn_in=burn_in,
        seed=seed,
    )
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
            mask[:: period[i], i] = 1

    elif "random_period" in config:
        choices = config["random_period"]["choices"]
        prob = config["random_period"]["prob"]

        period = np.random.choice(choices, N, p=prob)
        mask *= 0
        for i in range(N):
            mask[:: period[i], i] = 1

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



def save_missing_data(
    data_interp,
    data_masked,
    mask,
    gc,
    dataset_type,
    p,
    T,
    seed,
    F=None,
    original_data=None,
    missing_prob=0.2,
    interp="zoh",
    output_dir=None,
):
    """Save one missing-data dataset variant and return the output path."""
    canonical_interp = normalize_missing_imputation_method(interp)
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)

    filename = build_missing_dataset_filename(
        dataset_type,
        p,
        T,
        seed,
        F=F,
        missing_prob=missing_prob,
        interp=canonical_interp,
    )

    save_dict = {
        "data_interp": data_interp.squeeze(-1).astype(np.float32),
        "data_masked": data_masked.astype(np.float32),
        "mask": mask.astype(np.float32),
        "gc": gc,
        "imputation_method": np.asarray(canonical_interp),
    }
    if original_data is not None:
        save_dict["original_data"] = original_data.astype(np.float32)

    output_path = os.path.join(output_dir, filename)
    np.savez(output_path, **save_dict)
    return output_path



def generate_and_save_missing_dataset_variants(
    dataset_type,
    p,
    T,
    seed,
    missing_prob,
    interp_methods=("zoh", "linear", "GP"),
    F=None,
    output_dir=None,
    lag=3,
    missing_config=None,
    **generate_kwargs,
):
    """Generate and save multiple imputation variants for the same missing mask."""
    saved_paths = {}
    for interp in interp_methods:
        canonical_interp = normalize_missing_imputation_method(interp)
        if dataset_type == "VAR":
            data_interp, data_masked, mask, gc, original_data = generate_missing_var(
                p=p,
                T=T,
                lag=lag,
                seed=seed,
                missing_config=missing_config,
                interp=canonical_interp,
                **generate_kwargs,
            )
        else:
            data_interp, data_masked, mask, gc, original_data = generate_missing_lorenz_96(
                p=p,
                T=T,
                F=F,
                seed=seed,
                missing_config=missing_config,
                interp=canonical_interp,
                **generate_kwargs,
            )

        saved_paths[canonical_interp] = save_missing_data(
            data_interp,
            data_masked,
            mask,
            gc,
            dataset_type=dataset_type,
            p=p,
            T=T,
            seed=seed,
            F=F,
            original_data=original_data,
            missing_prob=missing_prob,
            interp=canonical_interp,
            output_dir=output_dir,
        )
    return saved_paths



def interp_masked_data(data, mask, interp="zoh"):
    interp = normalize_missing_imputation_method(interp)
    if interp == "zoh":
        T, D = data.shape
        new_data = deepcopy(data)
        for i in range(1, T):
            new_data[i] = data[i] * mask[i] + new_data[i - 1] * (1 - mask[i])
        return new_data

    x = np.argwhere(mask > 0)[:, 0]
    func = interpolate.interp1d(
        x,
        data[x, :],
        kind=interp,
        axis=0,
        copy=True,
        fill_value="extrapolate",
    )
    new_x = np.arange(0, data.shape[0])
    return func(new_x)



def interp_multivar_data(data, mask, interp="zoh"):
    interp = normalize_missing_imputation_method(interp)
    if data.ndim == 2:
        data = data[..., np.newaxis]  # (T, p, 1)
        mask = mask[..., np.newaxis]  # (T, p, 1)

    if interp == "GP":
        return interp_multivar_with_gauss_process(data, mask)

    new_data = np.zeros_like(data)
    _, N, _ = data.shape
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
