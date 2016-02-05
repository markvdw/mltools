import time

import numpy as np
import numpy.random as rnd
import matplotlib.pyplot as plt

import mltools.optimise as mlopt
import mltools.drawtools as mldraw


def f(x):
    return 0.7*x[0]**2.0 + x[1]**2.0 - 0.3*x[0]*x[1]


def g(x):
    res = np.zeros(len(x))
    res[0] = 1.4*x[0] - 0.3*x[1]
    res[1] = 2.0*x[1] - 0.3*x[0]
    time.sleep(0.1)
    return res

mldraw.plot_2d_func(f, xstep=0.1, ystep=0.1)
plt.show()

xt = rnd.randn(2) * 1000
fd = mlopt.finite_difference(f, xt)
ds = mlopt.check_grad(f, g, x0=xt)
print("Maximum percentage difference: %.2f%%" % ds.percent_max)

# print mlopt.minimize(f, jac=g, x0=xt, method="l-bfgs-b")
r = mlopt.minimize(f, jac=g, x0=xt, method="scg")
