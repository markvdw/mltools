import numpy as np

def quad (func, lim_left, lim_right, delta=None):
    if delta is None:
        delta = (float(lim_right) - float(lim_left)) / 1000

    N = int(np.ceil(lim_right - lim_left) / delta)

    X = np.linspace(lim_left, lim_right, N)

    I = 0
    for x in X:
        I += delta * func(x)

    return (I, np.inf)
