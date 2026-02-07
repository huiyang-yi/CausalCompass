"""

Reference:
    [1] https://github.com/cloud36/graphical_granger_methods

"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV


def run_lgc_simple(X, args, seed):
    alphas = args.lgc_alphas

    n_vars = X.shape[1]
    data = pd.DataFrame(X, columns=[f'x{i}' for i in range(n_vars)])

    causal_matrix = np.zeros((n_vars, n_vars))

    for target_idx in range(n_vars):
        target_var = f'x{target_idx}'
        predictor_vars = [f'x{i}' for i in range(n_vars) if i != target_idx]

        X_features, y_target = create_lagged_features(data, target_var, predictor_vars, args.tau_max)

        lasso_cv = LassoCV(alphas=alphas, random_state=seed)
        lasso_cv.fit(X_features, y_target)

        feature_idx = 0
        all_vars = [target_var] + predictor_vars

        for var_idx, var in enumerate(all_vars):
            lag_coeffs = lasso_cv.coef_[feature_idx:feature_idx + args.tau_max]

            max_abs_coef = np.max(np.abs(lag_coeffs))

            if max_abs_coef > args.threshold:
                if var == target_var:
                    causal_matrix[target_idx, target_idx] = 1
                else:
                    orig_var_idx = int(var[1:])
                    causal_matrix[target_idx, orig_var_idx] = 1

            feature_idx += args.tau_max

    return causal_matrix

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