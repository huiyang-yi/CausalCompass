import torch
import numpy as np
import tempfile
import shutil
from types import SimpleNamespace
from ..base import BaseCausalAlgorithm

class CUTS(BaseCausalAlgorithm):
    """
    CUTS is a deep learning-based causal discovery method tailored for irregular time series data.

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
    device : str, default 'cuda'
        Computation device

    Examples
    --------
    >>> from causalcompass.algorithms import CUTS
    >>> model = CUTS(input_step=10, batch_size=32, weight_decay=0.001, device='cuda')
    >>> predicted_adj = model.run(X, true_cm=true_adj, mask=mask, original_data=X)
    >>> all_metrics, no_diag_metrics = model.eval(true_adj, predicted_adj)
    """
    def __init__(self, input_step=10, batch_size=32, weight_decay=0.001, device='cuda', seed=None, **kwargs):
        """
        Initialize CUTS
        """
        super().__init__(seed=seed)
        self._eval_output_type = "continuous"
        self.device = device
        self.config_params = {
            'cuts_input_step': input_step,
            'cuts_batch_size': batch_size,
            'cuts_weight_decay': weight_decay,
            'device': device
        }

    def run(self, X, true_cm, mask=None, original_data=None):
        """
        Run CUTS algorithm.

        :param X: Interpolated or complete time series data, shape (T, p).
        :param mask: Data mask, shape (T, p). Defaults to all 1s.
        :param original_data: Original complete data (if available). Defaults to X.
        :param true_cm: True causal matrix.
        :return: Predicted adjacency matrix, shape (p, p).
        """
        # Default handling
        if mask is None: mask = np.ones_like(X)
        if original_data is None: original_data = X

        # Mock args
        run_params = self.config_params.copy()
        _, p = X.shape
        run_params['p'] = p
        args_mock = SimpleNamespace(**run_params)

        # Import internal modules
        from .cuts_main import CUTS as CUTSModel
        from .utils.logger import MyLogger
        from .cuts_config import create_cuts_config

        opt = create_cuts_config(args_mock)

        if self.device.startswith("cuda") and not torch.cuda.is_available():
            device = torch.device("cpu")
        else:
            device = torch.device(self.device)

        temp_dir = tempfile.mkdtemp(prefix="cuts_temp_")
        log = MyLogger(log_dir=temp_dir, stdout=False, stderr=False, tensorboard=True)

        try:
            # Reshape data to (T, p, 1) as expected by CUTS
            # Check dimensions to avoid double expansion
            data_interp = X[:, :, np.newaxis] if X.ndim == 2 else X
            orig_data = original_data[:, :, np.newaxis] if original_data.ndim == 2 else original_data
            data_mask = mask[:, :, np.newaxis] if mask.ndim == 2 else mask

            cuts_model = CUTSModel(opt, log, device=device)
            
            # Train and get prediction
            predicted_adj = cuts_model.train(data_interp, data_mask, orig_data, true_cm)

            return predicted_adj

        finally:
            try:
                shutil.rmtree(temp_dir)
            except:
                pass
