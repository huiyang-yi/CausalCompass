import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV

def compute_lgc_score_matrix(X, args, seed):
    """
    Compute the maximum absolute Lasso coefficient matrix across lags.
    """
    alphas = args.lgc_alphas

    n_vars = X.shape[1]
    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_vars)])

    score_matrix = np.zeros((n_vars, n_vars), dtype=float)

    for target_idx in range(n_vars):
        target_var = f'x{target_idx}'
        predictor_vars = [f'x{i}' for i in range(n_vars) if i != target_idx]

        X_features, y_target = create_lagged_features(data, target_var, predictor_vars, args.tau_max)

        lasso_cv = LassoCV(alphas=alphas, random_state=seed)
        lasso_cv.fit(X_features, y_target)

        feature_idx = 0
        all_vars = [target_var] + predictor_vars

        for var in all_vars:
            lag_coeffs = lasso_cv.coef_[feature_idx:feature_idx + args.tau_max]
            max_abs_coef = np.max(np.abs(lag_coeffs))

            if var == target_var:
                score_matrix[target_idx, target_idx] = max_abs_coef
            else:
                orig_var_idx = int(var[1:])
                score_matrix[target_idx, orig_var_idx] = max_abs_coef

            feature_idx += args.tau_max

    return score_matrix

def threshold_lgc_score_matrix(score_matrix, threshold):
    return (np.asarray(score_matrix) > threshold).astype(int)

def run_lgc_simple(X, args, seed):
    score_matrix = compute_lgc_score_matrix(X, args, seed)
    return threshold_lgc_score_matrix(score_matrix, args.threshold)

def create_lagged_features(data, target_col, predictor_cols, max_lag):
    n_samples = len(data)
    start_idx = max_lag
    n_valid_samples = n_samples - start_idx

    all_vars = [target_col] + predictor_cols
    n_features = len(all_vars) * max_lag

    X = np.zeros((n_valid_samples, n_features))
    y = data[target_col].values[start_idx:]

    feature_idx = 0
    for var in all_vars:
        for lag in range(1, max_lag + 1):
            X[:, feature_idx] = data[var].values[start_idx - lag:n_samples - lag]
            feature_idx += 1

    return X, y
