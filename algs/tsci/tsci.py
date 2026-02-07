"""

Reference:
    [1] https://github.com/KurtButler/tangentspaces

"""
import numpy as np
import scipy  # type: ignore
import torch
from tqdm import tqdm
import torch.nn.functional as F
from jaxtyping import Float
from numpy import ndarray
import typing
from bmi.estimators import KSGEnsembleFirstEstimator


def tsci_nn(
    x_state: Float[ndarray, "T Q_x"],  # noqa: F722
    y_state: Float[ndarray, "T Q_y"],  # noqa: F722
    dx_state: Float[ndarray, "T Q_x"],  # noqa: F722
    dy_state: Float[ndarray, "T Q_y"],  # noqa: F722
    fraction_train: float = 0.8,
    lib_length: int = -1,
    use_mutual_info=False,
) -> typing.Tuple[Float[ndarray, "T 1"], Float[ndarray, "T 1"]]:  # noqa: F722
    """Performs Tangent Space Causal Inference (TSCI) with the nearest neighbors approach.

    Args:
        x_state (Float[ndarray, &quot;T Q_x&quot;]): delay embedding of signal $
        y_state (Float[ndarray, &quot;T Q_y&quot;]): delay embedding of signal $y$
        dx_state (Float[ndarray, &quot;T Q_x&quot;]): vector field of `x_state`
        dy_state (Float[ndarray, &quot;T Q_y&quot;]): vector field of `y_state`
        fraction_train (float, optional): fraction of training data in train/test split. Defaults to 0.8.
        lib_length (int, optional): library length to test with. If negative, defaults to `fraction_train * len(x_state.shape[0])`. Defaults to -1.

    Returns:
        typing.Tuple[Float[ndarray, &quot;T 1&quot;], Float[ndarray, &quot;T 1&quot;]]: correlation coefficients for causal directions $X \\to Y$ and $Y \\to X$
    """
    Q_x = dx_state.shape[1]
    Q_y = dy_state.shape[1]
    N_samples = dx_state.shape[0]
    N_train = int(fraction_train * N_samples)

    if lib_length < 0:
        lib_length = N_train

    # the pushforward dx vectors should look like the dy vectors
    x_pushforward = np.zeros_like(dy_state[N_train:])

    # initialize a KDTree to do repeated nearest-neighbor lookup
    K = 3 * Q_x
    kdtree = scipy.spatial.KDTree(x_state[:lib_length])

    ########################
    ## Pushforward X -> Y ##
    ########################
    # For each point in the test set, we find the Jacobian and pushfoward the corresponding `dx_state` sample
    for n in tqdm(range(N_train, x_state.shape[0]), leave=True, disable=True):
        # Query points and get displacement vectors
        _, ids = kdtree.query(x_state[n, :], K)
        x_tangents = x_state[ids, :] - x_state[n, :]
        y_tangents = y_state[ids, :] - y_state[n, :]

        # The Jacobian is the least-squares solution mapping x displacements to y displacements
        lstsq_results = scipy.linalg.lstsq(x_tangents, y_tangents)
        J = lstsq_results[0]

        # Pushforward is a vector-Jacobian product
        x_pushforward[n - N_train, :] = dx_state[n, :] @ J

    ########################
    ## Pushforward Y -> X ##
    ########################
    # For each point in the test set, we find the Jacobian and pushfoward the corresponding `dy_state` sample
    y_pushforward = np.zeros_like(dx_state[N_train:])
    K = 3 * Q_y

    kdtree = scipy.spatial.KDTree(y_state[:lib_length])
    for n in tqdm(range(N_train, y_state.shape[0]), leave=True, disable=True):
        # Query points and get displacement vectors
        _, ids = kdtree.query(y_state[n, :], K)
        x_tangents = x_state[ids, :] - x_state[n, :]
        y_tangents = y_state[ids, :] - y_state[n, :]

        # The Jacobian is the least-squares solution mapping y displacements to x displacements
        lstsq_results = scipy.linalg.lstsq(y_tangents, x_tangents)
        J = lstsq_results[0]

        # Pushforward is a vector-Jacobian product
        y_pushforward[n - N_train, :] = dy_state[n, :] @ J

    ########################
    # Compute correlations #
    ########################
    if use_mutual_info:
        score_x2y = KSGEnsembleFirstEstimator(neighborhoods=(10,)).estimate(
            dx_state[N_train:], y_pushforward
        )

        score_y2x = KSGEnsembleFirstEstimator(neighborhoods=(10,)).estimate(
            dy_state[N_train:], x_pushforward
        )
    else:
        dotprods = np.sum(dx_state[N_train:] * y_pushforward, axis=1)
        mags1 = np.sum(dx_state[N_train:] * dx_state[N_train:], axis=1)
        mags2 = np.sum(y_pushforward * y_pushforward, axis=1)
        score_x2y = dotprods / np.sqrt(mags1 * mags2 + 1e-16)

        dotprods = np.sum(dy_state[N_train:] * x_pushforward, axis=1)
        mags1 = np.sum(dy_state[N_train:] * dy_state[N_train:], axis=1)
        mags2 = np.sum(x_pushforward * x_pushforward, axis=1)
        score_y2x = dotprods / np.sqrt(mags1 * mags2 + 1e-16)

    return score_x2y, score_y2x


def tsci_torch(
    x_state: Float[ndarray, "T Q_x"],  # noqa: F722
    y_state: Float[ndarray, "T Q_y"],  # noqa: F722
    dx_state: Float[ndarray, "T Q_x"],  # noqa: F722
    dy_state: Float[ndarray, "T Q_y"],  # noqa: F722
    model_x2y: torch.nn.Module,
    model_y2x: torch.nn.Module,
    fraction_train: float = 0.8,
    device: str = "cpu",
    lam: float = 1e-4,
    max_epochs: int = 100,
    regularizer: typing.Tuple[typing.Callable, typing.Callable] = (
        lambda: 0.0,
        lambda: 0.0,
    ),
    verbose: bool = False,
    lr=2e-3,
) -> typing.Tuple[Float[ndarray, "T 1"], Float[ndarray, "T 1"]]:  # noqa: F722
    """Performs Tangent Space Causal Inference (TSCI) where the cross map is given by an MLP.

    Args:
        x_state (Float[ndarray, &quot;T Q_x&quot;]): delay embedding of signal $
        y_state (Float[ndarray, &quot;T Q_y&quot;]): delay embedding of signal $y$
        dx_state (Float[ndarray, &quot;T Q_x&quot;]): vector field of `x_state`
        dy_state (Float[ndarray, &quot;T Q_y&quot;]): vector field of `y_state`
        model_x2y (torch.nn.Module): torch module for the cross map from `x_state` to `y_state`
        model_y2x (torch.nn.Module): torch module for the cross map from `y_state` to `x_state`
        fraction_train (float, optional): fraction of training data in train/test split. Defaults to 0.8.
        device (str, optional): device to pass to torch. Defaults to "cpu".

    Returns:
        typing.Tuple[Float[ndarray, &quot;T 1&quot;], Float[ndarray, &quot;T 1&quot;]]: correlation coefficients for causal directions $X \\to Y$ and $Y \\to X$
    """
    # Inputs will be numpy arrays, so convert to torch
    X_tensor = torch.tensor(x_state).float().to(device)
    Y_tensor = torch.tensor(y_state).float().to(device)
    dX_tensor = torch.tensor(dx_state).float().to(device)
    dY_tensor = torch.tensor(dy_state).float().to(device)

    # Train/test split
    N_samples = len(X_tensor)
    N_train = int(fraction_train * N_samples)
    X_tensor_train = X_tensor[:N_train]
    X_tensor_val = X_tensor[N_train:]
    Y_tensor_train = Y_tensor[:N_train]
    Y_tensor_val = Y_tensor[N_train:]
    dX_tensor_val = dX_tensor[N_train:]
    dY_tensor_val = dY_tensor[N_train:]

    # Make datasets and datalaoders
    dset_X_train = torch.utils.data.TensorDataset(X_tensor_train)
    dset_X_val = torch.utils.data.TensorDataset(X_tensor_val)
    dset_Y_train = torch.utils.data.TensorDataset(Y_tensor_train)
    dset_Y_val = torch.utils.data.TensorDataset(Y_tensor_val)
    dl_X = torch.utils.data.DataLoader(dset_X_train, batch_size=32, shuffle=False)
    dl_Y = torch.utils.data.DataLoader(dset_Y_train, batch_size=32, shuffle=False)
    dl_val_X = torch.utils.data.DataLoader(dset_X_val, batch_size=32, shuffle=False)
    dl_val_Y = torch.utils.data.DataLoader(dset_Y_val, batch_size=32, shuffle=False)

    # The training loop in each direction is essentially the same, so we store relevant quantities in dictionaries and re-use training code.
    dls = {"X": dl_X, "Y": dl_Y}
    dls_val = {"X": dl_val_X, "Y": dl_val_Y}
    models = {
        "X": model_x2y.to(device),
        "Y": model_y2x.to(device),
    }
    val_tensors = {"X": X_tensor_val, "Y": Y_tensor_val}
    val_deriv_tensors = {"X": dX_tensor_val, "Y": dY_tensor_val}
    regularizers = {"X": regularizer[0], "Y": regularizer[1]}

    last_val_loss = -10_000.0
    N_since_improvement = 0

    res = []
    for side in [("X", "Y"), ("Y", "X")]:
        loss_criterion = torch.nn.MSELoss()
        model = models[side[0]]
        model.train()
        if verbose:
            print(f"Training MLP {side[0]}->{side[1]} with COCOB and LIPMLP")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)  #
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max_epochs * len(dls[side[0]])
        )
        for epoch in range(max_epochs):
            # Training step
            train_loss = 0.0
            model.train()
            for i, b in enumerate(zip(dls[side[0]], dls[side[1]])):
                optimizer.zero_grad()
                y_hat = model(b[0][0])
                loss = loss_criterion(y_hat, b[1][0]) + lam * regularizers[side[0]]()
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_loss += loss.detach()
            train_loss /= i + 1

            if verbose and ((epoch % 20) == 0 or epoch == max_epochs - 1):
                print(f"Training loss at epoch {epoch}: {train_loss}")

            # Validation step
            val_loss = 0.0
            model.eval()
            for i, b in enumerate(zip(dls_val[side[0]], dls_val[side[1]])):
                y_hat = model(b[0][0])
                loss = loss_criterion(y_hat, b[1][0]) + lam * regularizers[side[0]]()
                val_loss += loss.detach()
            val_loss /= i + 1

            # Track number of epochs since val improvement for early stopping
            if val_loss > last_val_loss:
                N_since_improvement += 1
            else:
                N_since_improvement = 0

            last_val_loss = val_loss

            if verbose and (
                (epoch % 20) == 0 or epoch == max_epochs - 1 or N_since_improvement > 3
            ):
                print(f"Validation loss at epoch {epoch}: {val_loss}")

            # Early stopping if validation loss has not improved
            if N_since_improvement > 3:
                break

        # Pushforward operation
        # Calculate Jacobian-vector product based on the MLP
        model.eval()
        _, pf = torch.autograd.functional.jvp(
            model, val_tensors[side[0]], v=val_deriv_tensors[side[0]][:, :]
        )
        res.append(
            F.cosine_similarity(val_deriv_tensors[side[1]], pf).cpu().detach().numpy()
        )

    # `res[0]` is from the cross-map of $X$ to $Y$, which corresponds to the causal direction $Y \\to X$.
    # Flip the direction here accordingly.
    return res[1], res[0]
