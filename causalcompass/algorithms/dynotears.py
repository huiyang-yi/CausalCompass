import numpy as np
import pandas as pd
from .base import BaseCausalAlgorithm

class DyNotears(BaseCausalAlgorithm):
    """
    Score-based causal discovery method that extends the NOTEARS framework to dynamic (time series) settings using continuous optimization with acyclicity constraints.

    References
    ----------
    https://github.com/mckinsey/causalnex

    Parameters
    ----------
    tau_max : int, default 3
        Maximum time lag
    wthre : float, default 0.01
        Weight threshold for edge filtering
    lambda_w : float, default 0.1
        Parameter for L1 regularization of intra-slice edges
    lambda_a : float, default 0.1
        Parameter for L1 regularization of inter-slice edges
    max_iter : int, default 100
        Max number of dual ascent steps during optimization
    h_tol : float, default 1e-8
        Tolerance for acyclicity constraint

    Examples
    --------
    >>> from causalcompass.algorithms import DyNotears
    >>> model = DyNotears(tau_max=3, wthre=0.01, lambda_w=0.1, lambda_a=0.1)
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, tau_max=3, wthre=0.01, lambda_w=0.1, lambda_a=0.1, max_iter=100, h_tol=1e-8, seed=None):
        """
        Initialize DYNOTEARS
        """
        super().__init__(seed=seed)
        self.tau_max = tau_max
        self.wthre = wthre
        self.lambda_w = lambda_w
        self.lambda_a = lambda_a
        self.max_iter = max_iter
        self.h_tol = h_tol

    def run(self, X):
        """
        Run DYNOTEARS algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        raw_result = self.run_raw(X)
        return self._threshold_from_raw(raw_result, self.wthre)

    def run_raw(self, X):
        """
        Run DYNOTEARS once and return the aggregated absolute edge-weight matrix.
        """
        from .causalnex.structure.dynotears import from_pandas_dynamic

        df = pd.DataFrame(X)
        sm = from_pandas_dynamic(
            df,
            p=self.tau_max,
            w_threshold=0.0,
            lambda_w=self.lambda_w,
            lambda_a=self.lambda_a,
            max_iter=self.max_iter,
            h_tol=self.h_tol,
        )

        p = X.shape[1]
        score_matrix = np.zeros((p, p), dtype=float)

        for c, e, weight in sm.edges.data("weight"):
            source_var = int(c.split('_')[0])
            target_var = int(e.split('_')[0])
            score_matrix[target_var, source_var] = max(
                score_matrix[target_var, source_var],
                abs(float(weight)),
            )

        return {
            'raw_type': 'score_matrix',
            'score_matrix': score_matrix,
        }

    def _threshold_from_raw(self, raw_result, threshold):
        return (raw_result['score_matrix'] > threshold).astype(int)
