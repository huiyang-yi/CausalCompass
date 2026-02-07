"""

Reference:
    [1] https://github.com/cloud36/graphical_granger_methods
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def run_var_simple(X, args):
    """
    Run VAR-based Granger causality analysis using coefficient magnitudes.
    """
    # Get parameters
    max_lag = getattr(args, 'tau_max', 5)
    threshold = getattr(args, 'threshold', 0.1)

    T, p = X.shape

    # Convert to DataFrame with datetime index
    df = pd.DataFrame(X, columns=[f'X{i}' for i in range(p)])
    df.index = pd.date_range(start='2000-01-01', periods=T, freq='D')

    # Fit VAR model
    res = VAR(df).fit(max_lag)

    # Get parameters excluding intercept and take absolute values
    pred = np.abs(res.params[1:])

    # Reshape parameters to (p, p, max_lag) format
    pred_tensor = np.stack([
        pred.values[:, x].reshape(max_lag, p).T
        for x in range(pred.shape[1])
    ])

    # Take maximum across lags
    predicted_adj = np.max(pred_tensor, axis=2)

    # Apply threshold to get binary matrix
    predicted_adj = (predicted_adj > threshold).astype(int)

    # Set diagonal to 1
    np.fill_diagonal(predicted_adj, 1)

    return predicted_adj
