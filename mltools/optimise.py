import sys
import collections
import time

import numpy as np
import scipy.optimize as opt

import matplotlib.pyplot as plt


def minimize(fun,
             x0,
             args=(),
             method=None,
             jac=None,
             hess=None,
             hessp=None,
             bounds=None,
             constraints=(),
             tol=None,
             callback=None,
             options=None,
             hist=None,
             timeout=np.inf,
             optscale=1,
             resume_trace=None):
    hist = optimisation_history(fun, jac, chaincallback=callback, optscale=optscale, args=args, resume_trace=resume_trace)

    fun_timeout = create_timeout_function(fun, timeout)

    start_time = time.time()
    try:
        r = opt.minimize(fun_timeout,
                         x0,
                         args=args,
                         method=method,
                         jac=jac,
                         hess=hess,
                         hessp=hessp,
                         bounds=bounds,
                         constraints=constraints,
                         tol=tol,
                         callback=hist.iteration,
                         options=options)
    except (OptimisationTimeout, KeyboardInterrupt) as e:
        if type(e) is OptimisationTimeout:
            message = "Optimisation timeout after %fs" % (time.time() - start_time)
        elif type(e) is KeyboardInterrupt:
            message = "Optimisation cancelled by keyboard after %fs" % (time.time() - start_time)
        else:
            raise NotImplementedError("Shouldn't happen!")
        if len(hist.hist.x_hist) > 0:
            final_x = hist.hist.x_hist[-1]
        else:
            final_x = x0
        r = opt.OptimizeResult(x=final_x,
                               success=False,
                               message=message,
                               fun=fun(final_x, *args),
                               jac=jac(final_x, *args))

    finalf = fun(r.x, *args)
    finalg = jac(r.x, *args)
    hist.hist.add_calc(time.time() - hist.time_offset, finalf * optscale, finalg * optscale)
    print("")
    r.hist = hist

    return r


class OptimisationTrace(object):
    def __init__(self, x_hist=None, iter_times=None):
        if x_hist is None:
            self.x_hist = []
            self.iter_times = []
        else:
            if len(x_hist) == len(iter_times):
                raise ValueError("Need times for iterations")
            self.x_hist = x_hist
            self.iter_times = iter_times

        self.calc_funcs = []
        self.calc_funcs_times = []
        self.calc_grads = []
        self.calc_grads_times = []

    def iteration(self, x, t):
        self.x_hist.append(x)
        self.iter_times.append(t)

    def add_calc(self, t, f=None, g=None):
        if f is not None:
            self.calc_funcs.append(f)
            self.calc_funcs_times.append(t)

        if g is not None:
            self.calc_grads.append(np.mean(g**2.0)**0.5)
            self.calc_grads_times.append(t)

    def calc(self, f, g=None, iterations=None):
        if iterations is None:
            iterations = np.arange(0, len(self.iter_times))

        calc_funcs = []
        calc_funcs_times = []

        search_start = 0
        for i in iterations:
            found, search_start = self._find_previous_calc(self.iter_times[i], self.calc_funcs_times, search_start)
            if found:
                calc_funcs.append(self.calc_funcs[search_start])
            else:
                calc_funcs.append(f(self.x_hist[i]))
            calc_funcs_times.append(self.iter_times[i])

        self.calc_funcs = calc_funcs
        self.calc_funcs_times = calc_funcs_times

    @staticmethod
    def _find_previous_calc(req_time, times, start_iter):
        while start_iter < len(times) and times[start_iter] < req_time:
            if times[start_iter] == req_time:
                return True, start_iter
            start_iter += 1
        return False, start_iter


class optimisation_history (object):
    def __init__(self, func, grad, args=(), print_gap=1.0, print_growth=1.2, print_max=100, verbose=1, chaincallback=None, optscale=1, resume_trace=None):
        if chaincallback is not None:
            self.chaincallback = chaincallback
        else:
            self.chaincallback = lambda *x: 0

        # Storage of parameters of the optimisation
        self.func = func
        self.grad = grad
        self.func_args = args
        self.optscale = optscale

        # Parameters for the display during optimisation
        self.init_print_gap = print_gap
        self.print_growth = print_growth
        self.print_max = print_max
        self.verbose = verbose

        # Static variables for use during optimisation
        self.i = 0
        self.nexti = self.init_print_gap
        self.print_gap = self.init_print_gap
        self.last_time = time.time()

        # Variables for final display
        if resume_trace is None:
            self.hist = OptimisationTrace()
            self.time_offset = time.time()
        else:
            self.hist = resume_trace
            self.time_offset = time.time() - resume_trace.iter_times[-1]

        print("Iter\tfunc\t\tgrad\t\titer/s\tTimestamp")

    def iteration(self, f, force_print=False):
        reltime = time.time() - self.time_offset
        self.hist.iteration(f, reltime)
        self.i += 1

        if self.verbose == 0:
            return

        if self.i == int(self.nexti) or force_print:
            cur_time = time.time()
            time_per_iter = int(self.print_gap) / (cur_time - self.last_time)

            fval = self.func(f, *self.func_args) * self.optscale
            gval = self.grad(f, *self.func_args) * self.optscale
            # print "%i\t%e\t%e\t%f" % (self.i, fval, np.sum(gval ** 2.0), time_per_iter)
            sys.stdout.write("\r")
            sys.stdout.write("%i\t%e\t%e\t%f\t%s\t" % (self.i, fval, np.sqrt(np.mean(gval ** 2.0)), time_per_iter, time.ctime()))
            sys.stdout.flush()
            # print("%i\t%e\t%e\t%f\r" % (self.i, fval, np.sqrt(np.sum(gval ** 2.0)), time_per_iter))

            self.print_gap = min(self.print_growth * self.print_gap, self.print_max)
            self.nexti += int(self.print_gap)
            self.last_time = cur_time

            self.hist.add_calc(reltime, fval, gval)
            self.chaincallback(self, f, fval, gval)
        else:
            # sys.stdout.write("%i\r" % self.i)
            # sys.stdout.flush()
            pass

    # def reset(self):
    #     self.i = 0
    #     self.nexti = self.init_print_gap
    #     self.print_gap = self.init_print_gap
    #     self.hist = []
    #     self.time_offset = 0
    #     self.resume()

    def resume(self):
        self.last_time = time.time()
        self.time_offset = time.time() - self.hist.iter_times[-1]
        print("Iter\tfunc\t\tgrad\t\titer/s")

    def update_hist(self, update_grad=False):
        # We only store the parameters at each iteration, not the actual objective function value. So now we need to
        # recompute it.
        func_append = []
        grad_append = []
        for i in xrange(len(self.func_hist), len(self.hist)):
            func_append.append(self.func(self.hist[i], *self.func_args) * self.optscale)

        if update_grad:
            for i in xrange(len(self.grad_hist), len(self.hist)):
                grad_append.append(
                    np.mean((self.grad(self.hist[i], *self.func_args) * self.optscale)**2.0)**0.5
                )

        self.func_hist = np.hstack((self.func_hist, func_append))
        assert len(self.hist) == len(self.func_hist)
        if update_grad:
            self.grad_hist = np.hstack((self.grad_hist, grad_append))
            assert len(self.hist) == len(self.grad_hist)

    def plot_f_hist(self, plot_grad=False, plot_opts={}, x_axis="iter"):
        linestyle = '-'
        axs = []

        # self.update_hist(plot_grad)
        self.hist.calc(lambda x: self.func(x, *self.func_args)*self.optscale)

        if x_axis == "iter":
            x = np.arange(0, len(self.hist.calc_funcs_times))
            xlabel = "Iteration"
        elif x_axis == "time":
            x = self.hist.calc_funcs_times
            xlabel = "Time (s)"
        else:
            raise NotImplementedError("Don't know this x-axis type")

        if plot_grad:
            axs.append(plt.subplot(2,1,1))
        else:
            axs.append(plt.gca())
        plt.plot(x, self.hist.calc_funcs, linestyle, **plot_opts)
        plt.xlabel(xlabel)
        plt.ylabel('Function value')
        if plot_grad:
            axs.append(plt.subplot(2,1,2))
            plt.plot(self.hist.calc_grads_times, self.hist.calc_grads, **plot_opts)

        return axs

    def plot_x_hist(self, prop=True, plot_opts={}, x_axis="iter", varlbls=None, slice=np.s_[:]):
        hist = np.array(self.hist)
        if x_axis == "iter":
            x = np.arange(0, hist.shape[0])
            xlabel = "iteration"
        elif x_axis == "time":
            x = np.array(self.times)
            xlabel = "time (s)"
        else:
            raise NotImplementedError("Don't know this x-axis type...")

        if prop:
            hist = hist / hist[-1, :]

        for i, h in enumerate(hist.T[slice, :]):
            plt.plot(x, h, label=varlbls[slice][i])
        plt.xlabel(xlabel)


class OptimisationTimeout(Exception):
    pass


def create_timeout_function(f, timeout, verbose=False):
    start_time = time.time()
    def tf(x, *args):
        if (time.time() - start_time) < timeout:
            return f(x, *args)
        else:
            if verbose:
                print("Timeout!")
            # return np.zeros(len(x))
            raise OptimisationTimeout("Optimisation has taken too long...")

    return tf


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
    """
    fd_sensitivity
    Parameters:
     fun     : Function to be tested
     x0      : Location to be tested
     cg      : Gradient at the location
     args    : Additional arguments to the function
     drange  : Logarithmic range to test on
     # n     : Number of locations
     plot_all: Plot all the changes
    """
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
            bck = x0[idx]
            x0[idx] += d
            f1 = fun(x0, *args)
            x0[idx] = bck
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
    try:
        if args is None:
            args = ()

        if jac is None:
            raise ValueError('Must supply a value for the Jacobian (gradient)')

        options_default = {'max_eps': .1,
                           'verbosity' :0,
                           'maxiter': -1,
                           'streak': 10,
                           'momentum': 0.0,
                           'min_eps':0}

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
            gprop = jac(xprop, *args)

            if (fprop < fold) and not (np.any(np.isnan(gprop))):
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
            elif eps < options['min_eps']:
                print('Finished due to not moving...')
                break

            if options['verbosity'] >= 4:
                print streak, x, grad, fold, eps
            if options['verbosity'] >= 3:
                # print '%i\tstreak: %i\tfold: %e\teps: %e\t sum(grad**2): %e\r' % (opt_iteration, streak, fold, eps, np.mean(grad**2.0)**0.5)
                print '%i\tstreak: %i\tfold: %e\teps: %e\t \r' % (opt_iteration, streak, fold, eps)
            elif options['verbosity'] >= 2:
                # print streak, fold, eps
                sys.stdout.write('streak: %i\tfold: %e\teps: %e\t sum(grad**2): %e\r' % (streak, fold, eps, np.mean(grad**2.0)**0.5))

            if callback != None:
                callback(x)

            opt_iteration += 1
    except KeyboardInterrupt:
        print("Finished due to keyboard interrupt...")

    return x