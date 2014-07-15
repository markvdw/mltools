###############################################################################
# prob.y
# Some probability functions used for Machine Learning programming in Python.
#
# Mark van der Wilk (mv310@cam.ac.uk)
###############################################################################

from __future__ import division

import numpy as np
import numpy.linalg as linalg
import numpy.random as random
from scipy import constants

class ProbDistBase(object):
    def logpdf(self):
        raise NotImplementedError()

    def logjpdf(self):
        raise NotImplementedError()

    def pdf(self):
        raise NotImplementedError()

    def sample(self):
        raise NotImplementedError()

    def entropy(self):
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

class MultivariateNormal(ProbDistBase):
    def __init__(self, mu, S, cS=None, iS=None):
        if type(S) is int:
            self.D = 1
            self.S = np.array([[S]])
        else:
            S = np.array(S)
            self.D = S.shape[0]

        self.mu = mu
        self._S = S
        if iS is None:
            self._iS = linalg.inv(S)
        else:
            self._iS = iS

        if cS is None:
            self._cS = linalg.cholesky(S)
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

    @property
    def S(self):
        return self._S

    @S.setter
    def S(self, value):
        self._S = value
        self._cS = linalg.cholesky(S)
        self._iS = linalg.inv(S)

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


def mvn_entropy(mu, Sigma):
    D = len(mu)

    _, logdet = linalg.slogdet(Sigma)
    return 0.5*D*(1.+np.log(2*np.pi)) + 0.5*logdet
