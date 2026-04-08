import numpy as np
from types import SimpleNamespace
from ..base import BaseCausalAlgorithm

class VAR(BaseCausalAlgorithm):
    """
    Granger causality-based causal discovery method that fits a Vector Autoregressive model and infers causal relationships from the estimated coefficient matrix.

    References
    ----------
    https://github.com/cloud36/graphical_granger_methods

    Parameters
    ----------
    tau_max : int, default 3
        Maximum time lag
    threshold : float, default 0.01
        Coefficient threshold for edge filtering

    Examples
    --------
    >>> from causalcompass.algorithms import VAR
    >>> model = VAR(tau_max=3, threshold=0.01)
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, tau_max=3, threshold=0.01, seed=None, **kwargs):
        """
        Initialize VAR
        """
        super().__init__(seed=seed)
        self.tau_max = tau_max
        self.threshold = threshold
        
        self.config_params = {
            'tau_max': tau_max,
            'threshold': threshold,
        }

    def run(self, X):
        """
        Run VAR algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        raw_result = self.run_raw(X)
        return self._threshold_from_raw(raw_result, self.threshold)

    def run_raw(self, X):
        args_mock = SimpleNamespace(**self.config_params)
        from .var_main import compute_var_score_matrix

        return {
            'raw_type': 'score_matrix',
            'score_matrix': compute_var_score_matrix(X, args_mock),
        }

    def _threshold_from_raw(self, raw_result, threshold):
        from .var_main import threshold_var_score_matrix

        return threshold_var_score_matrix(raw_result['score_matrix'], threshold)
