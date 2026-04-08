import torch
import numpy as np
from .base import BaseCausalAlgorithm
from .ngc.cmlp import cMLP as _cMLP
from .ngc.cmlp import train_model_ista


class CMLP(BaseCausalAlgorithm):
    """
    Deep learning-based causal discovery method that uses component-wise MLPs to model nonlinear Granger causality.

    References
    ----------
    https://github.com/iancovert/Neural-GC

    Parameters
    ----------
    lag : int, default 3
        Maximum time lag
    hidden_dim : list, default [100]
        Number of hidden units per layer
    lam : float, default 0.005
        Sparsity penalty term parameter
    lr : float, default 0.01
        Learning rate
    max_iter : int, default 50000
        Maximum training iterations
    lam_ridge : float, default 1e-2
        Ridge regularization parameter
    penalty : str, default 'H'
        Penalty type: 'GL' (group lasso), 'GSGL' (group sparse group lasso), or 'H' (hierarchical)
    device : str, default 'cuda'
        Computation device

    Examples
    --------
    >>> from causalcompass.algorithms import CMLP
    >>> model = CMLP(lag=3, hidden_dim=[100], lam=0.005, lr=0.01, max_iter=50000, device='cuda')
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """

    def __init__(self, lag=3, hidden_dim=[100], lam=0.005, lr=0.01, max_iter=50000, lam_ridge=1e-2, penalty='H',
                 device='cuda', seed=None):
        """
        Initialize cMLP
        """
        
        super().__init__(seed=seed)
        self._eval_output_type = "continuous"
        self.lag = lag
        self.hidden_dim = hidden_dim
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter
        self.lam_ridge = lam_ridge
        self.penalty = penalty
        self.device = device

    def run(self, X):
        """
        Run cMLP algorithm.

        :param X: Time series data, shape (T, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        # Device setup
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA is not available, switching to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(self.device)

        # Prepare data: Add batch dimension (1, T, p)
        # Note: cMLP expects torch tensor
        X_tensor = torch.tensor(X[np.newaxis], dtype=torch.float32, device=device)

        # Initialize model
        model = _cMLP(
            num_series=X.shape[-1],
            lag=self.lag,
            hidden=self.hidden_dim,
        ).to(device)

        # Train model
        train_model_ista(
            model,
            X_tensor,
            lam=self.lam,
            lam_ridge=self.lam_ridge,
            lr=self.lr,
            penalty=self.penalty,
            max_iter=self.max_iter,
            verbose=0,
        )

        predicted_adj = model.GC(threshold=False).cpu().data.numpy()

        return predicted_adj
