import torch
import numpy as np
import tempfile
import shutil
from types import SimpleNamespace
from ..base import BaseCausalAlgorithm

class CUTSPlus(BaseCausalAlgorithm):
    """
    CUTS+ extends the original CUTS framework.

    References
    ----------
    https://github.com/jarrycyx/UNN

    Parameters
    ----------
    input_step : int, default 10
        Number of past time steps used as input
    batch_size : int, default 32
        Training batch size
    weight_decay : float, default 0.001
        Controls the strength of regularization
    lambda_s : float, default 0.01
        Graph sparsity coefficient used during graph discovery
    device : str, default 'cuda'
        Computation device

    Examples
    --------
    >>> from causalcompass.algorithms import CUTSPlus
    >>> model = CUTSPlus(input_step=10, batch_size=32, weight_decay=0.001, lambda_s=0.01, device='cuda')
    >>> predicted_adj = model.run(X, true_cm=true_adj, mask=mask)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, input_step=10, batch_size=32, weight_decay=0.001, lambda_s=0.01, device='cuda', seed=None, **kwargs):
        """
        Initialize CUTS+
        """
        super().__init__(seed=seed)
        self._eval_output_type = "continuous"
        self.device = device
        self.config_params = {
            'cutsplus_input_step': input_step,
            'cutsplus_batch_size': batch_size,
            'cutsplus_weight_decay': weight_decay,
            'cutsplus_lambda_s': lambda_s,
            'device': device
        }

    def run(self, X, true_cm, mask=None):
        """
        Run CUTS+ algorithm.

        :param X: Time series data, shape (T, p).
        :param mask: Data mask, shape (T, p). 1 for observed, 0 for missing. Defaults to all 1s.
        :param true_cm: True causal matrix (p, p).
        :return: Predicted adjacency matrix, shape (p, p).
        """
        # Handle defaults
        if mask is None:
            mask = np.ones_like(X)

        # Construct a mock args object
        run_params = self.config_params.copy()
        _, p = X.shape
        run_params['p'] = p
        args_mock = SimpleNamespace(**run_params)

        from .cuts_plus import main as cutsplus_main
        from .utils.logger import MyLogger
        from .cutsplus_config import create_cutsplus_config

        # Create configuration
        opt = create_cutsplus_config(args_mock)

        # Device handling
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            print("CUDA not available, switching to CPU.")
            device = torch.device("cpu")
        else:
            device = torch.device(self.device)

        temp_dir = tempfile.mkdtemp(prefix="cutsplus_temp_")
        log = MyLogger(log_dir=temp_dir, stdout=False, stderr=False, tensorboard=False)

        try:
            # Run the algorithm
            predicted_adj = cutsplus_main(
                data=X,
                mask=mask,
                true_cm=true_cm,
                opt=opt,
                log=log,
                device=device
            )
            return predicted_adj

        finally:
            # Clean up temporary directory
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
