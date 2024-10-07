import numpy as np


def is_floating(x) -> bool:
    return isinstance(x, (float, np.float32, np.float64))


def is_array(x) -> bool:
    return isinstance(x, np.ndarray)
