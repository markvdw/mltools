###############################################################################
# prob.py
# Some probability functions used for Machine Learning programming in Python.
#
# Mark van der Wilk (mv310@cam.ac.uk)
###############################################################################

from __future__ import division

import collections
import operator

import numpy as np
import numpy.linalg as linalg
import numpy.random as random
from scipy import constants
from scipy.special import erf
import scipy.stats as stats

import matplotlib.pyplot as plt

import linalg as mlin

class ProbDistBase(object):
    """
    ProbDistBase
    Base class for a probability distribution. Maybe a better name for this should be "random variable". The idea is to
    have a completely encapsulated random variable that can be sampled from, and have its density evaluated. Should take
    any of its parameters during construction.
    """
    def logpdf(self, *args, **kwargs):
        raise NotImplementedError()

    def logjpdf(self, *args, **kwargs):
        return np.sum(self.logpdf(*args, **kwargs))

    def pdf(self, *args, **kwargs):
        return np.exp(self.logpdf(*args, **kwargs))

    def sample(self):
        raise NotImplementedError()

    def entropy(self):
        raise NotImplementedError()

    def entropy_mc(self, samples=1):
        s = self.sample(samples)
        return -self.logjpdf(s) / len(s)

    def plot(self):
        raise NotImplementedError()


class LikelihoodBase(object):
    """
    LikelihoodBase
    Base class for a likelihood. Should pass the data and any parameters during the construction so that the likelihood
    can then be evaluated with no further information. Should also be able to sample data given a parameter or sample
    from a model given a prior distribution.
    """
    def loglik(self, *args, **kwargs):
        raise NotImplementedError()

    def lik(self, *args, **kwargs):
        return np.exp(self.loglik(*args, **kwargs))

    def logpdf(self, *args, **kwargs):
        raise NotImplementedError()

    def logjpdf(self, *args, **kwargs):
        return np.sum(self.logpdf(*args, **kwargs))

    def pdf(self, *args, **kwargs):
        return np.exp(self.logpdf(*args, **kwargs))

    def sample(self):
        raise NotImplementedError()
    
        
class DummyImproperUniform(ProbDistBase):
    """
    DummyImproperUniform
    Convenient way to remove the influence of a particular variable from a
    joint probability calculation.
    """
    def logpdf(self, X):
        return 0.0

    def logjpdf(self, X):
        return 0.0

    def pdf(self, X):
        return 1.0

class Mixture(ProbDistBase):
    def __init__(self, distribution_list, weights):
        self._distlist = distribution_list
        self._weights = weights

    def logpdf(self, X):
        return np.log(np.sum(np.hstack([p.pdf(X) for p in self._distlist]) * self._weights[None, :], 1))
        

class MixtureOfGaussians(Mixture):
    def __init__(self, param_dist_list, weights):
        if type(param_dist_list) is list:
            Mixture.__init__(self, param_dist_list, weights)
        else:
            Mixture.__init__(self, [MultivariateNormal(*param) for param in param_list], weights)

        dims = set([dist.D for dist in self._distlist])
        if len(dims) > 1:
            raise ValueError("Can not have mixtures of different dimension...")
        self.D = self._distlist[0].D
        self.wp = stats.rv_discrete(values=(range(0, len(weights)), weights))

    def sample(self, N=1):
        c = collections.Counter(self.wp.rvs(size=N))
        samples = None
        for mixture_idx in c:
            s = self._distlist[mixture_idx].sample(c[mixture_idx])
            if samples is None:
                samples = s
            else:
                samples = np.vstack((s, samples))
        return samples

    def plot(self):
        if self.D == 1:
            min_p = self._distlist[np.argmin([p.mu for p in self._distlist])]
            max_p = self._distlist[np.argmax([p.mu for p in self._distlist])]
            X = np.linspace(min_p.mu - 4.0 * min_p.S.flatten()**0.5,
                            max_p.mu + 4.0 * max_p.S.flatten()**0.5,
                            500)[:, None]
            probs = self.pdf(X)
            plt.plot(X, probs)
            # print("Area under curve: %f" % (np.sum(probs) * (X[1] - X[0])))
    
    @classmethod
    def random_init(cls, D=2, M=3, meanscale=1.0):
        return cls([MultivariateNormal.random_init(D, meanscale) for _ in xrange(M)], random.dirichlet([1] * M))

    
class MultivariateNormal(ProbDistBase):
    def __init__(self, mu, S, cS=None, iS=None, jitchol=False):
        if (type(S) is int) or (type(S) is float):
            self.D = 1
            self.S = np.array([[S]])
        else:
            self.S = np.array(S)
            self.D = self.S.shape[0]

        self.mu = mu
        if iS is None:
            self._iS = linalg.inv(self.S)
        else:
            self._iS = iS

        if cS is None:
            if jitchol:
                self._cS = mlin.jit_chol(5)
            else:
                self._cS = linalg.cholesky(self.S)
        else:
            self._cS = cS

    def logpdf(self, X):
        return mvnlogpdf_p(X, self.mu, self._iS)

    def logjpdf(self, X):
        return np.sum(self.logpdf(X))

    def pdf(self, X):
        return mvnpdf_p(X, self.mu, self._iS)

    def sample(self, N=1):
        return self._cS.dot(random.randn(self.D, N)).T + self.mu

    def entropy(self):
        logdet_cov = 2.0*np.sum(np.log(np.diag(self._cS)))
        return 0.5 * self.D * (1. + np.log(2*np.pi)) + 0.5 * logdet_cov

    def plot(self):
        if self.D == 1:
            X = np.linspace(-4*self.S.flatten()**0.5 + self.mu, 4*self.S.flatten()**0.5 + self.mu, 100)[:, None]
            p = self.pdf(X)
            plt.plot(X, p)
            plt.title('mean: %f, std: %f' % (self.mu, self.S.flatten()**.5))
        elif self.D == 2:
            s = self.sample(300)
            plt.plot(s[:, 0], s[:, 1], 'x')
        elif self.D == 3:
            s = self.sample(300)
            plt.plot(s[:, 0], s[:, 1], s[:, 2], 'x')
        else:
            print "I don't know how to draw this. D = %i" % self.D

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, value):
        self._S = value
        self._cS = linalg.cholesky(value)
        self._iS = linalg.inv(value)

    @classmethod
    def random_init(cls, D=2, meanscale=1.0):
        cov = random.randn(D, D)
        return cls(random.randn(D) * meanscale, np.dot(cov, cov.T))

class Interval(object):
    def __init__(self, lower=0.0, upper=0.0):
        """
        Interval constructor

        Args:
            lower: Lower bound
            upper: Upper bound
        """
        self._lower = lower
        self._upper = upper
        self._range = upper - lower
        assert self._range >= 0.0

    @property
    def range(self):
        return self._range

    @property
    def lower(self):
        return self._lower

    @lower.setter
    def lower(self, value):
        self._lower = value
        self._range = self._upper - value

    @property
    def upper(self):
        return self._upper

    @upper.setter
    def upper(self, value):
        self._upper = value
        self._range = self._upper - self._lower

    def inside(self, x):
        return (x >= self._lower) and (x <= self.upper)

class MultivariateUniform(ProbDistBase):
    def __init__(self, hyper_rectangle):
        """
        MultivariateUniform constructor

        Args:
            hyper_rectangle: Array of Interval objects specifying the range of
                             the uniform distribution.
        """
        self._r = hyper_rectangle

        # Calculate density
        logdensity = 0
        for hr in hyper_rectangle:
            logdensity -= np.log(hr.range)

        self._ld = logdensity

    @property
    def D(self):
        return len(self._r)

    def logpdf(self, X):
        lpdf = np.empty(len(X))
        ninf = float('-inf')
        for n, x in enumerate(X):
            lpdf[n] = self._ld
            for val, i in zip(x, self._r):
                if not i.inside(val):
                    lpdf[n] = ninf
                    break
        return lpdf

    def pdf(self, X):
        return np.exp(self.logpdf(X))

    def sample(self, N=1):
        y = np.empty((N, self.D))

        for d, i in enumerate(self._r):
            for n in range(N):
                y[n, d] = random.uniform(i.lower, i.upper)

        return y

def _check_single_data(X, D):
    """
    Checks whether data given to one of the distributions is a single data
    point of dimension D, rather than a set of datapoints of dimension 1. This
    is necessary, as there is no distinction between row and column vectors in
    numpy (sadly).
    """

    # Alternatively, could use np.atleast_2d... Oh well.
    if (len(X.shape) == 1 and len(X) == D):
        return np.reshape(X, [1, D])
    else:
        return X

def mvnlogpdf_p (X, mu, PrecMat):
    """
    Multivariate Normal Log PDF

    Args:
        X      : NxD matrix of input data. Each ROW is a single sample.
        mu     : Dx1 vector for the mean.
        PrecMat: DxD precision matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = PrecMat.shape[0]
    X = _check_single_data(X, D)
    N = len(X)

    _, neglogdet = linalg.slogdet(PrecMat)
    normconst = -0.5 * (D * np.log(2 * constants.pi) - neglogdet)

    logpdf = np.empty((N, 1))
    for n, x in enumerate(X):
        d = x - mu
        logpdf[n] = normconst - 0.5 * d.dot(PrecMat.dot(d))

    return logpdf

def mvnlogpdf (X, mu, Sigma):
    """
    Multivariate Normal Log PDF

    Args:
        X    : NxD matrix of input data. Each ROW is a single sample.
        mu   : Dx1 vector for the mean.
        Sigma: DxD covariance matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = Sigma.shape[0]
    X = _check_single_data(X, D)
    N = len(X)

    _, logdet = linalg.slogdet(Sigma)
    normconst = -0.5 * (D * np.log(2 * constants.pi) + logdet)

    iS = linalg.inv(Sigma)
    logpdf = np.empty((N, 1))
    for n, x in enumerate(X):
        d = x - mu
        logpdf[n] = normconst - 0.5 * d.dot(iS.dot(d))

    return logpdf

def mvnpdf_p (X, mu, PrecMat):
    """
    Multivariate Normal PDF

    Args:
        X    : NxD matrix of input data. Each ROW is a single sample.
        mu   : Dx1 vector for the mean.
        Sigma: DxD precision matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = len(mu)
    X = _check_single_data(X, D)
    N = len(X)

    normconst = 1.0 / ((2*constants.pi)**(0.5*D)*linalg.det(PrecMat)**-0.5)

    pdf = np.empty((N, 1))
    for n, x in enumerate(X):
        d = x - mu
        pdf[n] = normconst * np.exp(-0.5 * d.dot(PrecMat.dot(d)))

    return pdf

def mvnpdf (X, mu, Sigma):
    """
    Multivariate Normal PDF

    Args:
        X    : NxD matrix of input data. Each ROW is a single sample.
        mu   : Dx1 vector for the mean.
        Sigma: DxD covariance matrix.

    Returns:
        Nx1 vector of log probabilities.
    """
    D = len(mu)
    X = _check_single_data(X, D)
    N = len(X)

    normconst = 1.0 / ((2*constants.pi)**(0.5*D)*linalg.det(Sigma)**0.5)

    pdf = np.empty((N, 1))
    iS = linalg.inv(Sigma)
    for n, x in enumerate(X):
        d = x - mu
        pdf[n] = normconst * np.exp(-0.5 * d.dot(iS.dot(d)))

    return pdf


def phi (z):
    """
    phi
    Calculate the univariate normal cumulative density function.
    """
    return 0.5 * (1 + erf(z / 2**0.5))


def mvn_entropy(mu, Sigma):
    D = len(mu)

    _, logdet = linalg.slogdet(Sigma)
    return 0.5*D*(1.+np.log(2*np.pi)) + 0.5*logdet
