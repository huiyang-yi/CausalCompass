import numpy as np
from types import SimpleNamespace
from ..base import BaseCausalAlgorithm

class LGC(BaseCausalAlgorithm):
    """
    Granger causality-based causal discovery method that augments the VAR model with a Lasso penalty term.

    References
    ----------
    https://github.com/cloud36/graphical_granger_methods

    Parameters
    ----------
    tau_max : int, default 3
        Maximum time lag
    threshold : float, default 0.01
        Edge filtering threshold
    lgc_alphas : list[float], default [1e-4, 5e-3, 1e-2, 2e-2, 5e-2]
        L1 regularization candidates

    Examples
    --------
    >>> from causalcompass.algorithms import LGC
    >>> model = LGC(tau_max=3, threshold=0.01, lgc_alphas=[1e-4, 5e-3, 1e-2, 2e-2, 5e-2])
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, tau_max=3, threshold=0.01, lgc_alphas=[1e-4, 5e-3, 1e-2, 2e-2, 5e-2], seed=None, **kwargs):
        """
        Initialize LGC
        """
        super().__init__(seed=seed)
        self.tau_max = tau_max
        self.threshold = threshold
        self.lgc_alphas = lgc_alphas
        
        self.config_params = {
            'tau_max': tau_max,
            'threshold': threshold,
            'lgc_alphas': lgc_alphas
        }

    def run(self, X):
        """
        Run LGC algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        raw_result = self.run_raw(X)
        return self._threshold_from_raw(raw_result, self.threshold)

    def run_raw(self, X):
        args_mock = SimpleNamespace(**self.config_params)

        from .lgc import compute_lgc_score_matrix

        return {
            'raw_type': 'score_matrix',
            'score_matrix': compute_lgc_score_matrix(X, args_mock, self.seed),
        }

    def _threshold_from_raw(self, raw_result, threshold):
        from .lgc import threshold_lgc_score_matrix

        return threshold_lgc_score_matrix(raw_result['score_matrix'], threshold)
