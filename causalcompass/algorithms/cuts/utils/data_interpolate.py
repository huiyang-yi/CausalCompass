from scipy import interpolate
import numpy as np
from copy import deepcopy


def interp_masked_data(data, mask, interp="zoh"):
    if interp not in ("zoh", "linear"):
        raise ValueError("Supported missing-data imputation methods: zoh, linear")
    if interp == "zoh":
        T, D = data.shape
        new_data = deepcopy(data)
        for i in range(1,T):
            new_data[i] = data[i]*mask[i] + new_data[i-1]*(1-mask[i])
        return new_data
    else:
        x = np.argwhere(mask>0)[:,0]
        func = interpolate.interp1d(x, data[x,:], kind=interp, 
                                    axis=0, copy=True, fill_value="extrapolate")
        new_x = np.arange(0, data.shape[0])
        return func(new_x)

def interp_multivar_data(data, mask, interp="zoh"):
    if interp not in ("zoh", "linear"):
        raise ValueError("Supported missing-data imputation methods: zoh, linear")
    new_data = np.zeros_like(data)
    _, N, _ = data.shape
    for node_i in range(N):
        new_data[:, node_i, :] = interp_masked_data(
            data[:, node_i, :], mask[:, node_i], interp=interp
        )
    return new_data
