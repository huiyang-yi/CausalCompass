import random
import torch
import numpy as np
from ..base import BaseCausalAlgorithm
from .nts_notears import NTS_NOTEARS, train_NTS_NOTEARS


class NTSNotears(BaseCausalAlgorithm):
    """
    NTS-NOTEARS is a score-based nonlinear extension of DYNOTEARS.

    References
    ----------
    https://github.com/xiangyu-sun-789/NTS-NOTEARS

    Parameters
    ----------
    tau_max : int, default 3
        Maximum time lag
    wthre : float or list, default 0.01
        Weight threshold for edge filtering
    lambda_1 : float, list, or str, default 0.001
        L1 regularization parameter
    lambda_2 : float, default 0.01
        L2 regularization parameter
    device : str, default 'cuda'
        Computation device
    max_iter : int, default 100
        Max number of dual ascent steps during optimization
    h_tol : float, default 1e-8
        Tolerance for acyclicity constraint
    rho_max : float, default 1e+16
        Maximum value for the augmented Lagrangian penalty parameter

    Examples
    --------
    >>> from causalcompass.algorithms import NTSNotears
    >>> model = NTSNotears(tau_max=3, wthre=0.01, lambda_1=0.001, lambda_2=0.01, device='cuda')
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """

    def __init__(self, tau_max=3, wthre=0.01, lambda_1=0.001, lambda_2=0.01, device='cuda', max_iter=100, h_tol=1e-8,
                 rho_max=1e+16, seed=None):
        """
        Initialize NTS-NOTEARS
        """
        super().__init__(seed=seed)
        self.tau_max = tau_max
        self.wthre = wthre
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.device = device
        self.max_iter = max_iter
        self.h_tol = h_tol
        self.rho_max = rho_max

    def run(self, X):
        """
        Run NTS-NOTEARS algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        raw_result = self.run_raw(X)
        return self._threshold_from_raw(raw_result, self.wthre)

    def run_raw(self, X):
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(self.seed)
                torch.cuda.manual_seed_all(self.seed)

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(self.device)

        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.double)

        try:
            X_double = X.astype(np.float64)
            _, d = X.shape

            model = NTS_NOTEARS(
                dims=[d, 2 * d, 1],
                number_of_lags=self.tau_max,
                variable_names_no_time=[f'X{i}' for i in range(d)]
            )

            if isinstance(self.lambda_1, list):
                lambda1_param = self.lambda_1
            elif isinstance(self.lambda_1, str) and ',' in self.lambda_1:
                lambda1_param = [float(x.strip()) for x in self.lambda_1.split(',')]
            else:
                lambda1_param = float(self.lambda_1)

            W_est_full = train_NTS_NOTEARS(
                model=model,
                normalized_X=X_double,
                device=device,
                lambda1=lambda1_param,
                lambda2=self.lambda_2,
                w_threshold=0.0,
                max_iter=self.max_iter,
                h_tol=self.h_tol,
                rho_max=self.rho_max,
                verbose=0,
            )

            p = d
            weights_to_targets = np.abs(W_est_full[:, -p:])
            score_matrix = np.zeros((p, p), dtype=float)

            for i in range(0, weights_to_targets.shape[0], p):
                score_matrix = np.maximum(score_matrix, weights_to_targets[i:i + p])

            return {
                'raw_type': 'score_matrix',
                'score_matrix': score_matrix,
            }
        finally:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            torch.set_default_dtype(previous_dtype)

    def _threshold_from_raw(self, raw_result, threshold):
        return (raw_result['score_matrix'] > threshold).astype(int).T
