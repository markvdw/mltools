###############################################################################
# Unit Tests
# Simple tests to confirm the consistancy between these functions here and the
# standard pdf functions for single variables.
###############################################################################
import unittest
import argparse
import time

import numpy as np
import numpy.random as rnd
import numpy.linalg as linalg

from scipy import stats

import theano
import theano.tensor as T

import prob as mlprob
import ThMultivariateNormal as thnorm

###############################################################################
# Parse the arguments...
###############################################################################
parser = argparse.ArgumentParser(description='Housekeeping for MLtools.py.')
parser.add_argument('-b', '--benchmark', help="Benchmark the different functions.", action="store_true")
parser.add_argument('-t', '--test', help="Run the unit tests.", action="store_true")
args = parser.parse_args()

class TestSequenceFunctions(unittest.TestCase):
    def setUp(self):
        self.tolerance = 10**-9

    def test_log_1d_eval(self):
        X = rnd.randn(100, 1)
        logpdf_tools = mlprob.mvnlogpdf(X, np.array([0]), np.array([[1]]))
        logpdf_sp = np.log(stats.norm.pdf(X))

        diff = np.sum(np.absolute(logpdf_tools - logpdf_sp)) / len(X)

        self.assertTrue(diff < self.tolerance)

    def test_exp_1d_eval(self):
        X = rnd.randn(100, 1)
        pdf_tools = mlprob.mvnpdf(X, [0], [[1]])
        pdf_sp = stats.norm.pdf(X)

        diff = np.sum(np.absolute(pdf_tools - pdf_sp)) / len(X)

        self.assertTrue(diff < self.tolerance)

    def test_log_diag(self):
        D = 100
        N = 3000
        X = rnd.randn(N, D)
        logpdf_tools = mlprob.mvnlogpdf(X, [0] * D, np.eye(D)).T
        logpdf_sp = np.log(np.prod(stats.norm.pdf(X), axis=1))

        diff = np.sum(np.absolute(logpdf_tools - logpdf_sp)) / N

        self.assertTrue(diff < self.tolerance)

    def test_exp_diag(self):
        D = 3
        N = 3000
        X = rnd.randn(N, D)
        pdf_tools = mlprob.mvnpdf(X, [0] * D, np.eye(D)).T
        pdf_sp = np.prod(stats.norm.pdf(X), axis=1)

        diff = np.sum(np.absolute(pdf_tools - pdf_sp)) / N

        self.assertTrue(diff < self.tolerance)

    def test_logpdf_vs_pdf(self):
        D = 9
        N = 100

        X = rnd.randn(N, D)
        S = rnd.randn(D, D)
        S = S.dot(S.T)
        logpdf = mlprob.mvnlogpdf(X, [0]*D, S)
        pdf = mlprob.mvnpdf(X, [0]*D, S)

        # logpdf vs pdf of mltools
        diff = np.sum(np.absolute(np.exp(logpdf) - pdf)) / len(X)
        self.assertTrue(diff < self.tolerance)

        # mltools vs theano logpdf
        thlogpdf = thnorm.logjpdf(X, [0]*D, S)
        diff = np.absolute(thlogpdf - np.sum(logpdf)) / len(X)
        self.assertTrue(diff < self.tolerance, "Difference : " + str(diff))

    def test_logpdf_precmat(self):
        D = 100
        N = 1000

        X = rnd.randn(N, D)
        S = rnd.randn(D, D)
        S = S.dot(S.T)
        logpdf = mlprob.mvnlogpdf_p(X, [0]*D, linalg.inv(S))
        logpdf_s = mlprob.mvnlogpdf(X, [0]*D, S)

        diff = np.sum(np.absolute(logpdf - logpdf_s)) / len(X)

        self.assertTrue(diff < self.tolerance)

        thlogpdf = thnorm.logjp_prec(X, [0] * D, linalg.inv(S))
        diff = (np.sum(logpdf) - thlogpdf) / len(X)

        self.assertTrue(diff < self.tolerance)

    def test_pdf_precmat(self):
        D = 100
        N = 1000

        X = rnd.randn(N, D)
        S = rnd.randn(D, D)
        S = S.dot(S.T)
        pdf = mlprob.mvnpdf_p(X, [0]*D, linalg.inv(S))
        pdf_s = mlprob.mvnpdf(X, [0]*D, S)

        diff = np.sum(np.absolute(pdf - pdf_s)) / len(X)

        self.assertTrue(diff < self.tolerance)

    def test_interval(self):
        l = rnd.randn()
        u = np.absolute(rnd.randn()) + l

        i = mlprob.Interval(l, u)

        self.assertAlmostEqual(i.range, u-l)

        l = rnd.randn()
        u = np.absolute(rnd.randn()) + l

        i.lower = l
        self.assertEqual(i.lower, l)
        i.upper = u
        self.assertEqual(i.upper, u)

        self.assertAlmostEqual(i.range, u-l)

    def test_mvnunif(self):
        homog = mlprob.MultivariateUniform([mlprob.Interval(-1.3, 1.3)] * 33)
        s = homog.sample(1000)
        self.assertLessEqual(np.max(s), 1.3)
        self.assertGreaterEqual(np.min(s), -1.3)

        hr = [mlprob.Interval(rnd.randn() - 20, rnd.randn() + 20) for _ in range(13)]
        logp = 0
        for r in hr:
            logp -= np.log(r.range)

        u = mlprob.MultivariateUniform(hr)
        self.assertAlmostEqual(logp, u._ld)

    def test_th_makelogpdf(self):
        D = 100
        N = 1000

        X = rnd.randn(N, D)
        S = rnd.randn(D, D)
        S = S.dot(S.T)
        mu = rnd.randn(D)
        pdf = np.sum(mlprob.mvnlogpdf_p(X, mu, S))

        #make_th_mvnlogjpdf(th_X, th_mu, th_prec):
        th_X = T.matrix('X')
        th_mu = T.vector('mu')
        th_prec = T.matrix('prec')
        th_logjpdf = thnorm.make_th_mvnlogjpdf(th_X, th_mu, th_prec)

        f = theano.function([th_X, th_mu, th_prec], th_logjpdf)
        
        logjpdf = f(X, mu, S)

        self.assertAlmostEqual(logjpdf, pdf)

        
    def test_mvn_entropy(self):
        D = [1, 2, 5, 10, 20, 50, 100]
        
        for d in D:
            mu = rnd.randn(d)
            S = rnd.randn(d, d)
            S = S.dot(S.T)
            pdf = mlprob.MultivariateNormal(mu, S)

            # Check for consitency
            self.assertAlmostEqual(pdf.entropy(), mlprob.mvn_entropy(mu, S))

            # Check for similarity to scipy
            self.assertAlmostEqual(pdf.entropy(), stats.multivariate_normal.entropy(mu, S))


if args.benchmark:
    print("Benchmarking...")

    D = 23
    N = 100000

    X = rnd.randn(N, D)
    S = rnd.randn(D, D)
    S = S.dot(S.T)

    # Benchmark mvnlogpdf
    start = time.time()
    logpdf = mlprob.mvnlogpdf(X, [0]*D, S)
    print(time.time() - start)

    # Benchmark mvnpdf
    start = time.time()
    pdf = mlprob.mvnpdf(X, [0]*D, S)
    print(time.time() - start)

    # Benchmark mvnlogpdf_p
    start = time.time()
    logpdf = mlprob.mvnlogpdf_p(X, [0]*D, linalg.inv(S))
    print(time.time() - start)

if args.test:
    suite = unittest.TestLoader().loadTestsFromTestCase(TestSequenceFunctions)
    unittest.TextTestRunner(verbosity=2).run(suite)
