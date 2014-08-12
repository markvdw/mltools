import numpy as np
import numpy.random as rnd

import simple_optimise as mlopt

vec = rnd.randn(10)
mat = rnd.randn(10, 10)
mat += mat.T


# Single output, single input
def f1_1(x):
    return x**2.0


def fD_1(x):
    return vec * x


def f1_D(x):
    return x.dot(mat.dot(x))


def f1_DD(x):
    return vec.dot(x.dot(vec))


def fDD_DD(x):
    return x * 3

fd1_1 = mlopt.finite_difference(f1_1, 3.0)
fdD_1 = mlopt.finite_difference(fD_1, 3.0)
fd1_D = mlopt.finite_difference(f1_D, rnd.randn(10))
fd1_DD = mlopt.finite_difference(f1_DD, rnd.randn(10, 10))
fdDD_DD = mlopt.finite_difference(fDD_DD, rnd.randn(10, 10))