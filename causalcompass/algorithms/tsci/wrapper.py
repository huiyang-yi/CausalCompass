import os
import numpy as np
from types import SimpleNamespace
from ..base import BaseCausalAlgorithm

class TSCI(BaseCausalAlgorithm):
    """
    Topology-based causal discovery method that leverages Takens’ state-space reconstruction theory to infer causality.

    References
    ----------
    https://github.com/KurtButler/tangentspaces

    Parameters
    ----------
    theta : float, default 0.5
        Parameter for lag selection
    fnn_tol : float, default 0.01
        Tolerance for the amount of false nearest neighbors
    seed : int, default None
        Random seed for reproducibility

    Examples
    --------
    >>> from causalcompass.algorithms import TSCI
    >>> model = TSCI(theta=0.5, fnn_tol=0.01, seed=0)
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, theta=0.5, fnn_tol=0.01, seed=None, **kwargs):
        """
        Initialize TSCI
        """
        super().__init__(seed=seed)
        self._eval_output_type = "continuous"
        
        self.config_params = {
            'tsci_theta': theta,
            'tsci_fnn_tol': fnn_tol,
        }

    def run(self, X):
        """
        Run TSCI algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        os.environ['JAX_PLATFORMS'] = 'cpu'

        args_mock = SimpleNamespace(**self.config_params)
        from .tsci_wrapper import run_tsci_simple
        predicted_adj = run_tsci_simple(X, args_mock, self.seed)
        
        return predicted_adj
