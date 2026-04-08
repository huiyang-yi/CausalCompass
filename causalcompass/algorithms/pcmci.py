import numpy as np
from .base import BaseCausalAlgorithm


class PCMCI(BaseCausalAlgorithm):
    """
    Constraint-based causal discovery method that combines the PC algorithm with Momentary Conditional Independence tests for time series data.

    References
    ----------
    https://github.com/jakobrunge/tigramite

    Parameters
    ----------
    tau_max : int, default 3
        Maximum time lag
    pc_alpha : float, default 0.05
        Significance level for PC stable step
    alpha : float, default 0.05
        Significance level for MCI test
    cond_ind_test : CondIndTest or None, default None
        Conditional independence test object from tigramite.independence_tests or an external test passed as a callable based on tigramite.independence_tests.CondIndTest. If None, ParCorr() will be used by default in run()

    Examples
    --------
    >>> from causalcompass.algorithms import PCMCI
    >>> model = PCMCI(tau_max=3, pc_alpha=0.05, alpha=0.05)
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """

    def __init__(self, tau_max=3, pc_alpha=0.05, alpha=0.05, cond_ind_test=None, seed=None):
        """
        Initialize PCMCI
        """
        super().__init__(seed=seed)
        self.tau_max = tau_max
        self.pc_alpha = pc_alpha
        self.alpha = alpha
        self.cond_ind_test = cond_ind_test

    def run(self, X):
        """
        Run PCMCI algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        raw_result = self.run_raw(X)
        return self._threshold_from_raw(raw_result, self.alpha)

    def run_raw(self, X):
        """
        Run PCMCI once and return the raw p-value matrix.
        """
        from tigramite.pcmci import PCMCI as TigramitePCMCI
        from tigramite.independence_tests.parcorr import ParCorr
        import tigramite.data_processing as pp

        dataframe = pp.DataFrame(X)
        cond_ind_test = self.cond_ind_test if self.cond_ind_test is not None else ParCorr()

        pcmci = TigramitePCMCI(dataframe=dataframe, cond_ind_test=cond_ind_test)
        results = pcmci.run_pcmci(
            tau_max=self.tau_max,
            pc_alpha=self.pc_alpha,
            alpha_level=self.alpha
        )

        return {
            'raw_type': 'p_matrix',
            'p_matrix': results['p_matrix'],
            'tau_max': self.tau_max,
        }

    def _threshold_from_raw(self, raw_result, threshold):
        p_matrix = raw_result['p_matrix']
        significant_matrix = p_matrix < threshold
        return np.any(significant_matrix[:, :, 1:], axis=2).astype(int).T
