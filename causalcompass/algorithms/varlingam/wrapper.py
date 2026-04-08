import numpy as np
from ..base import BaseCausalAlgorithm
from .var_lingam import VARLiNGAM as _VARLiNGAM

class VARLiNGAM(BaseCausalAlgorithm):
    """
    Noise-based causal discovery method that combines Vector Autoregressive model with Linear Non-Gaussian Acyclic Model (LiNGAM) for time series data.

    References
    ----------
    https://github.com/ckassaad/causal_discovery_for_time_series

    Parameters
    ----------
    tau_max : int, default 3
        Maximum time lag
    varlingamalpha : float, default 0.01
        Threshold for VARLiNGAM
    seed : int, default None
        Random seed for reproducibility

    Examples
    --------
    >>> from causalcompass.algorithms import VARLiNGAM
    >>> model = VARLiNGAM(tau_max=3, varlingamalpha=0.01, seed=0)
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, tau_max=3, varlingamalpha=0.01, seed=None):
        """
        Initialize VARLiNGAM
        """
        super().__init__(seed=seed)
        self.tau_max = tau_max
        self.varlingamalpha = varlingamalpha

    def run(self, X):
        """
        Run VARLiNGAM algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        raw_result = self.run_raw(X)
        return self._threshold_from_raw(raw_result, self.varlingamalpha)

    def run_raw(self, X):
        model = _VARLiNGAM(
            lags=self.tau_max, 
            criterion=None, 
            prune=True, 
            random_state=self.seed
        )
        
        model.fit(X)
        m = model.adjacency_matrices_
        p = X.shape[1]
        score_matrix = np.zeros((p, p), dtype=float)

        for lag in range(m.shape[0]):
            lag_scores = np.abs(m[lag])
            score_matrix = np.maximum(score_matrix, lag_scores)

        return {
            'raw_type': 'score_matrix',
            'score_matrix': score_matrix,
        }

    def _threshold_from_raw(self, raw_result, threshold):
        return (raw_result['score_matrix'] > threshold).astype(int).T
