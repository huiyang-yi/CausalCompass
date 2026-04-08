import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR

def compute_var_score_matrix(X, args):
    """
    Compute the maximum absolute VAR coefficient matrix across lags.
    """
    max_lag = getattr(args, 'tau_max', 5)

    T, p = X.shape

    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])
    df.index = pd.date_range(start='2000-01-01', periods=T, freq='D')

    res = VAR(df).fit(max_lag)
    pred = np.abs(res.params[1:])

    pred_tensor = np.stack([
        pred.values[:, x].reshape(max_lag, p).T
        for x in range(pred.shape[1])
    ])

    return np.max(pred_tensor, axis=2)

def threshold_var_score_matrix(score_matrix, threshold):
    predicted_adj = (np.asarray(score_matrix) > threshold).astype(int)
    np.fill_diagonal(predicted_adj, 1)
    return predicted_adj

def run_var_simple(X, args):
    """
    Run VAR-based Granger causality analysis using coefficient magnitudes.
    """
    threshold = getattr(args, 'threshold', 0.1)
    score_matrix = compute_var_score_matrix(X, args)
    return threshold_var_score_matrix(score_matrix, threshold)
