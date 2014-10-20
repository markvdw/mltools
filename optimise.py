import sys
import collections
import time

import numpy as np

import matplotlib.pyplot as plt


class optimisation_history (object):
    def __init__(self, func, grad, print_gap=1.0, print_growth=1.2, print_max=100):
        self.func = func
        self.grad = grad
        self.init_print_gap = print_gap
        self.print_growth = print_growth
        self.print_max = print_max

        self.i = 0
        self.nexti = self.init_print_gap
        self.print_gap = self.init_print_gap
        self.last_time = time.time()

        self.hist = []

    def iteration(self, f):
        self.hist.append(f)
        self.i += 1

        if self.i == int(self.nexti):
            cur_time = time.time()
            time_per_iter = int(self.print_gap) / (cur_time - self.last_time)

            fval = self.func(f)
            gval = self.grad(f)
            # print "%i\t%e\t%e\t%f" % (self.i, fval, np.sum(gval ** 2.0), time_per_iter)
            sys.stdout.write("%i\t%e\t%e\t%f\r" % (self.i, fval, np.sum(gval ** 2.0), time_per_iter))

            self.print_gap = min(self.print_growth * self.print_gap, self.print_max)
            self.nexti += int(self.print_gap)
            self.last_time = cur_time
        else:
            sys.stdout.write("%i\r" % self.i)
            sys.stdout.flush()

    def reset(self):
        self.i = 0
        self.nexti = self.init_print_gap
        self.print_gap = self.init_print_gap
        self.hist = []

        self.resume()

    def resume(self):
        self.last_time = time.time()
        print("Iter\tfunc\t\tgrad\t\titer/s")

    def plot_f_hist(self, start_iter=0):
        func_hist = []
        for f in self.hist:
            func_hist.append(self.func(f))

        iters = np.arange(start_iter, start_iter + len(func_hist))

        # plt.plot(iters, func_hist)
        plt.plot(iters, func_hist, 'x')
        plt.xlabel('Iteration')
        plt.ylabel('Function value')

        return len(func_hist) + start_iter


def fd_sensitivity_struct(fun, struct, changevar, cg, fun_args=None, drange=(-12, 0), n=60, plot_all=False):
    if fun_args is None:
        fun_args = ()

    dl = np.logspace(drange[0], drange[1], n)
    fdl = []
    for d in dl:
        fdl.append(finite_difference_struct(fun, struct, changevar, fun_args, d=d).flatten())
        sys.stdout.write('.')

    _plot_sensitivity(dl, fdl, cg, plot_all)

    return dl, fdl, cg


def fd_sensitivity(fun, x0, cg, args=None, drange=(-12, 0), n=60, plot_all=False):
    if args is None:
        args = ()

    dl = np.logspace(drange[0], drange[1], n)
    fdl = []
    for d in dl:
        fdl.append(finite_difference(fun, x0, args=args, d=d).flatten())
        sys.stdout.write('.')

    _plot_sensitivity(dl, fdl, cg, plot_all)

    return dl, fdl, cg


def _plot_sensitivity(dl, fdl, cg, plot_all=False):
    fdl = np.array(fdl) / cg.flatten()

    if plot_all:
        plt.semilogx(dl, fdl)
    else:
        plt.semilogx(dl, np.nanmin(fdl, 1), dl, np.nanmax(fdl, 1))

    # plt.semilogx(dl, [1.05] * len(dl))
    # plt.semilogx(dl, [0.95] * len(dl))
    plt.fill_between(dl, [1.05] * len(dl), [0.95] * len(dl), alpha=0.05)
    plt.fill_between(dl, [1.01] * len(dl), [0.99] * len(dl), alpha=0.15)

    plt.ylabel('estimated gradient')
    plt.xlabel('finite difference delta')
    plt.title('Sensitivity of finite difference')
    plt.ylim(0, 5)

    plt.grid()


def finite_difference_struct(fun, struct, changevar, fun_args=None, d=10**-4):
    if fun_args is None:
        fun_args = ()

    def wrapper(x, *args):
        changevar[:] = x
        return fun(struct, *args)

    x0 = changevar + 0
    fd = finite_difference(wrapper, x0, fun_args, d)
    changevar[:] = x0
    return fd


def grad_pd(cg, fd):
    pd = (cg - fd) / fd * 100.



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
        f0 = fun(x0, *args)
        result_shape = f0.shape

        fd = np.zeros(x0.shape + result_shape)
        for idx in np.ndindex(fd.shape[:len(x0.shape)]):
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
    pd = np.abs((fd - cg) / fd) * 100.

    percent_diff = np.nanmax(pd)
    max_diff = np.max(np.abs(fd - cg))
    loc_percent_max = np.unravel_index(np.nanargmax(pd), pd.shape)
    percent_diff_axis = np.nanmax(pd, 0)

    DiffStats = collections.namedtuple("DiffStats", "diff_max percent_max loc_percent_max percent_diff_axis pd")

    return DiffStats(diff_max=max_diff,
                     percent_max=percent_diff,
                     loc_percent_max=loc_percent_max,
                     percent_diff_axis=percent_diff_axis,
                     pd=pd)


def print_diffstats(diffstats, diff_max=True, percent_max=True, loc_percent_max=True, pd_axis=False):
    if diff_max:
        print "Maximum difference           : %e" % diffstats.diff_max
    if percent_max:
        print "Maximum percentage difference: %f" % diffstats.percent_max
    if loc_percent_max:
        print "Gradient shape               : " + str(diffstats.pd.shape)
        print "Location of max pd           : " + str(diffstats.loc_percent_max)
    if pd_axis:
        print "pd_axis                      : " + str(diffstats.percent_diff_axis)


def gradient_descent(fun, x0, jac=None, args=None, tol=10**-4, maxiter=-1, callback=None, options=None):
    if args is None:
        args = ()

    if jac is None:
        raise ValueError('Must supply a value for the Jacobian (gradient)')

    options_default = {'max_eps': .1,
                       'verbosity' :0,
                       'maxiter': -1,
                       'streak': 10,
                       'momentum': 0.0}

    if options is None:
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
    moved = True
    while opt_iteration != maxiter:
        if moved:
            grad = jac(x, *args)

        xprop = x - eps * grad + momentum
        fprop = fun(xprop, *args)

        if fprop < fold:
            x = xprop
            fold = fprop
            streak += 1

            momentum = momentum + momeps * - eps * grad

            if streak >= options['streak']:
                eps *= 2.0
                eps = min(eps, options['max_eps'])

            moved = True
        else:
            streak = 0
            momentum = 0.0
            eps /= 2.0
            moved = False

        # Termination condition
        if np.sum(grad ** 2) < tol:
            if options['verbosity'] >= 1:
                print('Finished due to tolerance...')
            break

        if options['verbosity'] >= 3:
            print streak, x, grad, fold, eps
        elif options['verbosity'] >= 2:
            # print streak, fold, eps
            sys.stdout.write('streak: %i\tfold: %e\teps: %e\t sum(grad**2): %e\r' % (streak, fold, eps, np.sum(grad**2.0)))

        if callback != None:
            callback(x)

        opt_iteration += 1

    return x
