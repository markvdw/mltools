import sys
import collections

import numpy as np


def finite_difference(fun, x0, args=None, d=10**-4):
    """
    finite_difference
    Calculates the finite difference of fun() around x0. fun() can return an
    array of any dimension. Input/parameter can also be of any dimension.
    :param fun: Function to calculate the fd of.
    :param x0: Location around which to calculate fd.
    :param args: Extra arguments to fun.
    :param d: Tolerance
    :return: The finite difference.
    """
    if args is None:
        args = ()

    if isinstance(x0, np.ndarray):
        result_shape = fun(x0).shape

        fd = np.zeros(x0.shape + result_shape)

        for idx in np.ndindex(fd.shape[:len(x0.shape)]):
            f0 = fun(x0, *args)
            x0[idx] += d
            f1 = fun(x0, *args)
            x0[idx] -= d
            fd[idx] = (f1 - f0) / d

        return fd
    else:
        return (fun(x0 + d, *args) - fun(x0, *args)) / d


def diffstats(fd, cg):
    """
    diffstats
    Returns different measures of the difference between two multidimensional
    matrices. Made for use with getting an idea what the difference is between
    a finite difference calculation and a calculated gradient
    :param fd: Matrix A
    :param cg: Matrix B
    :return:
    """
    percent_diff = np.nanmax(np.abs((fd - cg) / fd)*100.)
    max_diff = np.max(np.abs(fd - cg))

    DiffStats = collections.namedtuple("DiffStats", "diff_max percent_max")

    return DiffStats(diff_max=max_diff, percent_max=percent_diff)

def print_diffstats(d, diff_max=True, percent_max=True):
    if diff_max:
        print "Maximum difference           : %e" % d.diff_max
    if percent_max:
        print "Maximum percentage difference: %f" % d.percent_max


def gradient_descent(fun, x0, jac=None, args=None, tol=10**-4, maxiter=-1, callback=None, options=None):
    if (args == None):
        args = ()

    if (jac == None):
        raise ValueError('Must supply a value for the Jacobian (gradient)')

    options_default = {'max_eps':0.1,
                       'verbosity':0,
                       'maxiter':-1,
                       'streak':10,
                       'momentum':0.0}
    if (options == None):
        options = options_default

    for defkey in options_default:
        if defkey not in options:
            options[defkey] = options_default[defkey]
    
    x = x0
    eps = options['max_eps']
    momeps = options['momentum']

    fold = fun(x, *args)
    momentum = 0.0

    opt_iteration = 0
    streak = 0
    while (opt_iteration != maxiter):
        grad = jac(x, *args)

        xprop = x - eps * grad + momentum
        fprop = fun(xprop, *args)

        if (fprop < fold):
            x = xprop
            fold = fprop
            streak += 1

            momentum = momentum + momeps * - eps * grad

            if (streak >= options['streak']):
                eps *= 2.0
                eps = min(eps, options['max_eps'])
        else:
            streak = 0
            momentum = 0.0
            eps /= 2.0

        # Termination condition
        if (np.sum(grad ** 2) < tol):
            if (options['verbosity'] >= 1):
                print('Finished due to tolerance...')
            break

        if (options['verbosity'] >= 3):
            print streak, x, grad, fold, eps
        elif (options['verbosity'] >= 2):
            # print streak, fold, eps
            sys.stdout.write('streak: %i\tfold: %f\teps: %f\t sum(grad**2): %f\r' % (streak, fold, eps, np.sum(grad**2.0)))

        if (callback != None):
            callback(x)

        opt_iteration += 1

    return x
