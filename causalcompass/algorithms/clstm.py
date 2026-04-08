import torch
import numpy as np
from .base import BaseCausalAlgorithm
from .ngc.clstm import cLSTM as _cLSTM
from .ngc.clstm import train_model_ista

class CLSTM(BaseCausalAlgorithm):
    """
    Deep learning-based causal discovery method that uses component-wise LSTMs to model nonlinear Granger causality.

    References
    ----------
    https://github.com/iancovert/Neural-GC

    Parameters
    ----------
    context : int, default 10
        Context window length
    hidden_dim : int, default 100
        Number of units in LSTM cell
    lam : float, default 0.005
        Sparsity penalty term parameter
    lr : float, default 0.01
        Learning rate
    max_iter : int, default 20000
        Maximum training iterations
    lam_ridge : float, default 1e-2
        Ridge regularization parameter
    device : str, default 'cuda'
        Computation device

    Examples
    --------
    >>> from causalcompass.algorithms import CLSTM
    >>> model = CLSTM(context=10, hidden_dim=100, lam=0.005, lr=0.01, max_iter=20000, device='cuda')
    >>> predicted_adj = model.run(X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, context=10, hidden_dim=100, lam=0.005, lr=0.01, max_iter=20000, lam_ridge=1e-2, device='cuda', seed=None):
        """
        Initialize cLSTM
        """
        super().__init__(seed=seed)
        self._eval_output_type = "continuous"
        self.context = context
        self.hidden_dim = hidden_dim
        self.lam = lam
        self.lr = lr
        self.max_iter = max_iter
        self.lam_ridge = lam_ridge
        self.device = device

    def run(self, X):
        """
        Run cLSTM algorithm.

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
        X_tensor = torch.tensor(X[np.newaxis], dtype=torch.float32, device=device)

        # Initialize model
        model = _cLSTM(
            num_series=X.shape[-1],
            hidden=self.hidden_dim,
        ).to(device)

        # Train model
        train_model_ista(
            model, 
            X_tensor,
            context=self.context,
            lam=self.lam,
            lam_ridge=self.lam_ridge,
            lr=self.lr,
            max_iter=self.max_iter,
            verbose=0,
        )

        # Extract graph structure
        predicted_adj = model.GC(threshold=False).cpu().data.numpy()

        return predicted_adj
