import numpy as np
import scipy  # type: ignore
from jaxtyping import Float
from numpy import ndarray
from statsmodels.tsa import stattools  # type: ignore
from typing import Tuple
import warnings


def next_pow_two(n: int) -> int:
    """Gets the next power of two greater than `n`. Code from [1].

    [1] https://dfm.io/posts/autocorr/

    Args:
        n (int): number to get next power of two of

    Returns:
        int: next power of two greater than `n`
    """
    i = 1
    while i < n:
        i = i << 1
    return i


def autocorr_func_1d(
    x: Float[ndarray, " N"], norm: bool = True
) -> Float[ndarray, " N"]:
    """Computes the autocorrelation function (ACF) of a signal. Code from [1].

    [1] https://dfm.io/posts/autocorr/

    Args:
        x (Float[ndarray, &quot; N&quot;]): signal values
        norm (bool, optional): whether to normalize to autocorrelation. Defaults to True.

    Raises:
        ValueError: if `len(x.shape) > 1`

    Returns:
        Float[ndarray, &quot; N&quot;]: the ACF of `x`
    """
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")
    n = next_pow_two(len(x))

    # Compute the FFT and then (from that) the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n

    # Optionally normalize
    if norm:
        acf /= acf[0]

    return acf


def auto_window(taus: Float[ndarray, " N"], c: float) -> int:
    """Automated windowing procedure following Sokal (1989). Code from [1].

    [1] https://dfm.io/posts/autocorr/

    Args:
        taus (Float[ndarray, &quot; N&quot;]): autocorrelation times
        c (float): constant used in method

    Returns:
        int: window length
    """
    m = np.arange(len(taus)) < c * taus
    if np.any(m):
        return int(np.argmin(m))
    return len(taus) - 1


def autocorr_new(y: Float[ndarray, " N"], c=5.0) -> float:
    """Returns the autocorrelation time

    [1] https://dfm.io/posts/autocorr/

    Args:
        y (Float[ndarray, &quot; N&quot;]): _description_
        c (float, optional): _description_. Defaults to 5.0.

    Returns:
        float: _description_
    """
    y = np.atleast_2d(y)
    f = np.zeros(y.shape[1])
    for yy in y:
        f += autocorr_func_1d(yy)
    f /= len(y)
    taus = 2.0 * np.cumsum(f) - 1.0
    window = auto_window(taus, c)
    return taus[window]


def generate_lorenz_rossler(
    ts: Float[ndarray, " T"], z0: Float[ndarray, " 6"], C: float
) -> Tuple[Float[ndarray, "T 3"], Float[ndarray, "T 3"]]:
    """Generates data according to the popular coupled Lorenz-Rossler system.

    Args:
        ts (Float[ndarray, &quot; T&quot;]): times to integrate over.
        z0 (Float[ndarray, &quot; 6&quot;]): initial state.
        C (float): coupling parameter.

    Returns:
        Tuple[Float[ndarray, "T 3"], Float[ndarray, "T 3"]]: tuple of `X`, `Y`
    """

    def f(x, t):
        x_return = np.empty(6)
        # X System
        x_return[0] = -6 * (x[1] + x[2])
        x_return[1] = 6 * (x[0] + 0.2 * x[1])
        x_return[2] = 6 * (0.2 + x[2] * (x[0] - 5.7))

        # Y System
        x_return[3] = 10 * (-x[3] + x[4])
        x_return[4] = 28 * x[3] - x[4] - x[3] * x[5] + C * x[1] ** 2
        x_return[5] = x[3] * x[4] - 8 * x[5] / 3

        return x_return

    z = scipy.integrate.odeint(f, z0, ts)
    return z[:, :3], z[:, 3:]


def generate_rossler_rossler(
    ts: Float[ndarray, " T"],
    z0: Float[ndarray, " 6"],
    C: float,
    omega_1: float = 0.5,
    omega_2: float = 2.515,
) -> Tuple[Float[ndarray, "T 3"], Float[ndarray, "T 3"]]:
    """Generates data according to the popular coupled Lorenz-Rossler system.

    Args:
        ts (Float[ndarray, &quot; T&quot;]): times to integrate over.
        z0 (Float[ndarray, &quot; 6&quot;]): initial state.
        C (float): coupling parameter.

    Returns:
        Tuple[Float[ndarray, "T 3"], Float[ndarray, "T 3"]]: tuple of `X`, `Y`
    """

    def f(x, t):
        x_return = np.empty(6)
        # X System
        x_return[0] = -omega_1 * x[1] - x[2]
        x_return[1] = omega_1 * x[0] + 0.15 * x[1]
        x_return[2] = 0.2 + x[2] * (x[0] - 10)

        # Y System
        x_return[3] = -omega_2 * x[4] - x[5] + C * (x[0] - x[3])
        x_return[4] = omega_2 * x[3] + 0.72 * x[4]
        x_return[5] = 0.2 + x[5] * (x[3] - 10.0)

        return x_return

    z = scipy.integrate.odeint(f, z0, ts)
    return z[:, :3], z[:, 3:]


def false_nearest_neighbors(
    y: Float[ndarray, " T"],
    tau: int,
    fnn_tol: float = 0.01,
    Q_max: int = 20,
    rho: float = 17.0,
) -> int:
    """Computes a heuristic embedding dimension of `y` with time lag `tau` using the false nearest neighbors (FNN) algorithm.

    Args:
        y (Float[ndarray, &quot; T&quot;]): time series to compute FNN for.
        tau (int): time lag to use, for example as computed by `lag_select`.
        fnn_tol (float, optional): tolerance for the amount of false nearest neighbors. Defaults to 0.01.
        Q_max (int, optional): maximum allowed embedding dimension. Defaults to 20.
        rho (float, optional): magic number as proposed in the FNN paper. Defaults to 17.0.

    Returns:
        int: embedding dimension
    """
    Q = 1
    fnn_flag = False

    # Q is repeatedly increased until the number of false nearest neighbors falls below `fnn_tol`
    while not fnn_flag:
        Q += 1
        if Q > Q_max:
            warnings.warn("FNN did not converge.")
            return Q_max

        M1 = delay_embed(y, tau, Q)
        M2 = delay_embed(y, tau, Q + 1)

        M1 = M1[: M2.shape[0]]
        fnn = np.zeros(M1.shape[0])

        kdtree = scipy.spatial.KDTree(M1)

        for n in range(M1.shape[0]):
            _, ids = kdtree.query(M1[n, :], 2)
            # We may consider only the nearest neighbors, whose index is `ids[1]`
            Rd = np.linalg.norm(M1[ids[1], :] - M1[n, :], 2) / np.sqrt(Q)
            # Nearest neighbors will be much closer in the lower dimension
            # so ||M_2[n] - M_2[NN]||_2 / ||M_1[n] - M_1[NN]||_2 will be large
            fnn[n] = np.linalg.norm(M2[n, :] - M2[ids[1], :], 2) > rho * Rd

        if np.mean(fnn) < fnn_tol:
            fnn_flag = True

    return Q


def discrete_velocity(x: Float[ndarray, "T 1"], smooth=False) -> Float[ndarray, "T 1"]:
    """Gets the discrete derivative of a time series.
    This simply wraps `np.gradient`, so it uses a second order finite difference method
    everywhere except the boundaries.

    Args:
        x (Float[ndarray, &quot; T&quot;]): time series array

    Returns:
        Float[ndarray, &quot; T&quot;]: gradient
    """
    if smooth:
        return scipy.signal.savgol_filter(x, 5, 2, deriv=1, axis=0)
    else:
        return np.gradient(x, axis=0)


def lag_select(x: Float[ndarray, " T"], theta: float = 0.5, max_tau: int = 100) -> int:
    """Selects a time lag based on the autocorrelation function (ACF).

    Args:
        x (Float[ndarray, &quot; T&quot;]): the time series.
        theta (float, optional): the desired autocorrelation to fall under. Defaults to 0.5.
        max_tau (int, optional): maximum allowable time lag. Defaults to 100.

    Returns:
        int: selected time lag
    """
    # Calculate ACF, first on a default sub=timeseries
    acf = stattools.acf(x - x.mean())
    # Calculate for the entire time series if all values are about the threshold
    if np.all(acf >= theta):
        acf = stattools.acf(x, min(max_tau, len(x) - 1))

    tau = int(np.argmax(acf < theta))
    if tau == 0:
        tau = max_tau
    return tau


def delay_embed(
    x: Float[ndarray, "T 1"], lag: int, embed_dim: int
) -> Float[ndarray, "T embed_dim"]:
    """Computes the delay embedding with lag `tau` and dimension `embed_dim` of `x`

    Args:
        x (Float[ndarray, &quot; T&quot;]): time series
        lag (int): lag for delay embedding
        embed_dim (int): desired dimension

    Returns:
        Float[ndarray, &quot; T&quot;, &quot; embed_dim&quot;]: delay embedding of `x` with lag `tau` and embedding dimension `embed_dim`
    """
    num_x = x.shape[0] - (embed_dim - 1) * lag
    embed_list = []

    for i in range(embed_dim):
        embed_list.append(
            x[
                (embed_dim - 1) * lag
                - (i * lag) : (embed_dim - 1) * lag
                - (i * lag)
                + num_x
            ].reshape(-1, x.shape[1])
        )
    return np.concatenate(embed_list, axis=-1)
