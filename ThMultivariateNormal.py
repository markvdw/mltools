###############################################################################
# thprob.y
# Some probability functions used for Machine Learning programming in Python
# implemented using Theano.
#
# Mark van der Wilk (mv310@cam.ac.uk)
###############################################################################

import numpy as np

from scipy import constants as const

import theano
import theano.tensor as T
from theano.sandbox.linalg import ops as sT

th_X = T.matrix('mvnnorm_data')
th_mu = T.vector('mvnnorm_mu')
th_S = T.matrix('mvnnorm_sigma')
th_D = th_X.shape[1]
th_N = th_X.shape[0]

th_prec = sT.matrix_inverse(th_S)

th_d = th_X - th_mu

th_logjp = T.sum(-0.5*th_D*T.log(2*const.pi) +
                -0.5*T.log(sT.det(th_S)) +
                -0.5*sT.diag(T.dot(th_d, T.dot(th_prec, th_d.T)))
                )

th_logjp_stable = T.sum(-0.5*th_D*T.log(2*const.pi) +
                       -T.sum(T.log(sT.diag(sT.cholesky(th_S)))) +
                       -0.5*sT.diag(T.dot(th_d, T.dot(th_prec, th_d.T)))
                       )

th_logjp_prec = T.sum(-0.5*th_D*T.log(2*const.pi) +
                      0.5*T.log(sT.det(th_prec)) +
                     -0.5*sT.diag(T.dot(th_d, T.dot(th_prec, th_d.T))))

f_logjp = theano.function([th_X, th_mu, th_S], th_logjp)
f_logjp_stable = theano.function([th_X, th_mu, th_S], th_logjp_stable)

logjp_prec = theano.function([th_X, th_mu, th_prec], th_logjp_prec)
th_dlogjpdf_dmu = theano.grad(th_logjp, th_mu)
th_dlogjpdf_dS = theano.grad(th_logjp, th_S)
th_dlogjpdf_dX = theano.grad(th_logjp, th_X)

dlogjpdf_dmu = theano.function([th_X, th_mu, th_S], th_dlogjpdf_dmu)
dlogjpdf_dS = theano.function([th_X, th_mu, th_S], th_dlogjpdf_dS)
dlogjpdf_dX = theano.function([th_X, th_mu, th_S], th_dlogjpdf_dX)

def logjpdf(X, mu, S):
    p = f_logjp(X, mu, S)
    if (p == float('inf')) or (p == float('-inf')):
        print('stable')
        p = f_logjp_stable(X, mu, S)

    return p

def pdf(X, mu, S):
    return np.exp(logjpdf(X, mu, S))

class MultivariateNormal(object):
    def __init__(self, X=None, mu=None, S=None):
        self.X = X
        self.mu = mu
        self.S = S

    def logjpdf(self, X=None, mu=None, S=None):
        return logjpdf(*self._replace_none_params(X, mu, S))

    def jpdf(self, X=None, mu=None, S=None):
        return np.exp(self.logjpdf(X, mu, S))

    def sample(self, X=None, mu=None, S=None):
        raise NotImplementedError()

    def dmu(self, X=None, mu=None, S=None):
        return dlogjpdf_dmu(*self._replace_none_params(X, mu, S))

    def dS(self, X=None, mu=None, S=None):
        return dlogjpdf_dS(*self._replace_none_params(X, mu, S))

    def dX(self, X=None, mu=None, S=None):
        return dlogjpdf_dX(*self._replace_none_params(X, mu, S))

    def _replace_none_params(self, X, mu, S):
        if X is None:
            X = self.X
        if mu is None:
            mu = self.mu
        if S is None:
            S = self.S
        return (X, mu, S)