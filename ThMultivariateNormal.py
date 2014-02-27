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

th_logp = T.sum(-0.5*th_D*T.log(2*const.pi) +
                -0.5*T.log(sT.det(th_S)) +
                -0.5*sT.diag(T.dot(th_d, T.dot(th_prec, th_d.T)))
                )

th_logp_stable = T.sum(-0.5*th_D*T.log(2*const.pi) +
                       -T.sum(T.log(sT.diag(sT.cholesky(th_S)))) +
                       -0.5*sT.diag(T.dot(th_d, T.dot(th_prec, th_d.T)))
                       )

th_logp_prec = T.sum(-0.5*th_D*T.log(2*const.pi) +
                      0.5*T.log(sT.det(th_prec)) +
                     -0.5*sT.diag(T.dot(th_d, T.dot(th_prec, th_d.T))))

f_logp = theano.function([th_X, th_mu, th_S], th_logp)
f_logp_stable = theano.function([th_X, th_mu, th_S], th_logp_stable)

logp_prec = theano.function([th_X, th_mu, th_prec], th_logp_prec)
th_dlogpdf_dmu = theano.grad(th_logp, th_mu)
th_dlogpdf_dS = theano.grad(th_logp, th_S)
th_dlogpdf_dX = theano.grad(th_logp, th_X)

dlogpdf_dmu = theano.function([th_X, th_mu, th_S], th_dlogpdf_dmu)
dlogpdf_dS = theano.function([th_X, th_mu, th_S], th_dlogpdf_dS)
dlogpdf_dX = theano.function([th_X, th_mu, th_S], th_dlogpdf_dX)

def logpdf(X, mu, S):
    p = f_logp(X, mu, S)
    if (p == float('inf')) or (p == float('-inf')):
        print('stable')
        p = f_logp_stable(X, mu, S)

    return p

def pdf(X, mu, S):
    return np.exp(logpdf(X, mu, S))