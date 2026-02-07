"""

Reference:
    [1] https://github.com/KurtButler/tangentspaces

"""

import numpy as np
from typing import Optional, Tuple

from .utils import (
    lag_select,
    false_nearest_neighbors,
    delay_embed,
    discrete_velocity,
)
from .tsci import tsci_nn

def process_tsci_scores(r_x2y, r_y2x):

    r_x2y_clipped = np.maximum(r_x2y, 0)
    r_y2x_clipped = np.maximum(r_y2x, 0)

    causal_strength_x2y = np.mean(r_x2y_clipped)
    causal_strength_y2x = np.mean(r_y2x_clipped)

    return causal_strength_x2y, causal_strength_y2x

def run_tsci_simple(X: np.ndarray, args, seed: Optional[int] = None) -> np.ndarray:
    """
    Run TSCI algorithm following the source project approach.

    Parameters
    ----------
    X : np.ndarray, shape (T, p)
        Time series data
    args : argparse.Namespace
        Arguments containing TSCI hyperparameters
    seed : int, optional
        Random seed

    Returns
    -------
    predicted_adj : np.ndarray, shape (p, p)
        Predicted adjacency matrix
    """

    if seed is not None:
        np.random.seed(seed)

    # Extract hyperparameters with defaults matching source project
    theta = getattr(args, 'tsci_theta', 0.5)
    fnn_tol = getattr(args, 'tsci_fnn_tol', 0.005)

    T, p = X.shape
    predicted_adj = np.zeros((p, p))

    # Pairwise analysis following source project pattern
    for i in range(p):
        for j in range(i, p):
            x_signal = X[:, i].reshape(-1, 1)
            y_signal = X[:, j].reshape(-1, 1)

            # Get embedding hyperparameters (no artificial bounds)
            # tau_x = lag_select(x_signal, theta=theta)
            # tau_y = lag_select(y_signal, theta=theta)
            tau_x = min(lag_select(x_signal, theta=theta), 5)
            tau_y = min(lag_select(y_signal, theta=theta), 5)

            Q_x = false_nearest_neighbors(x_signal, tau_x, fnn_tol=fnn_tol)
            Q_y = false_nearest_neighbors(y_signal, tau_y, fnn_tol=fnn_tol)

            # Create delay embeddings
            x_state = delay_embed(x_signal, tau_x, Q_x)
            y_state = delay_embed(y_signal, tau_y, Q_y)

            # Truncate to same length (following source project: -100)
            truncated_length = min(x_state.shape[0], y_state.shape[0]) - 100

            x_state = x_state[-truncated_length:]
            y_state = y_state[-truncated_length:]

            # Get velocities with finite differences
            dx_dt = discrete_velocity(x_signal)
            dy_dt = discrete_velocity(y_signal)

            # Delay embed velocity vectors
            dx_state = delay_embed(dx_dt, tau_x, Q_x)
            dy_state = delay_embed(dy_dt, tau_y, Q_y)

            dx_state = dx_state[-truncated_length:]
            dy_state = dy_state[-truncated_length:]

            # Perform TSCI analysis
            r_x2y, r_y2x = tsci_nn(
                x_state,
                y_state,
                dx_state,
                dy_state,
                fraction_train=0.8,
                use_mutual_info=False,
            )

            strength_i2j, strength_j2i = process_tsci_scores(r_x2y, r_y2x)

            if i == j:
                predicted_adj[i, j] = strength_j2i
            else:
                predicted_adj[i, j] = strength_j2i  # j → i
                predicted_adj[j, i] = strength_i2j  # i → j

    return predicted_adj
