###############################################################################
# linalg.py
# Some useful linear algebra utilities. Maybe in the future a full library to
# do certain common transformations?
#
# Mark van der Wilk (mv310@cam.ac.uk)
###############################################################################

import math

import numpy as np

import scipy.linalg as linalg


def jit_chol(mat):
    try:
        return linalg.cholesky(mat, lower=True)
    except linalg.LinAlgError:
        I = np.eye(mat.shape[0])
        epsarr = np.logspace(-10, 0, 11)
        for eps in epsarr:
            print "Trying %e" % eps
            try:
                result = linalg.cholesky(mat + eps * I, lower=True)
                print "It worked!"
                return result
            except linalg.LinAlgError:
                # Go to the next iteration...
                pass
        raise linalg.LinAlgError("Matrix not posdef, even with jitter.")


def jitify(func, mat):
    """
    jitify
    Repeatedly try 'func' while catching LinAlgErrors, with increasing jitter.
    :param func:
    :return:
    """
    I = np.eye(mat.shape[0])
    epsarr = np.logspace(-11, 0, 12)
    epsarr[0] = 0.0
    for eps in epsarr:
        try:
            result = func(mat + eps*I)
            return result
        except linalg.LinAlgError:
            pass

    raise linalg.LinAlgError("Matrix not posdef, even with jitter.")


def check_sym(mat):
    if np.all(mat == mat.T):
        print("Completely symmetric!")
    else:
        pd = (mat - mat.T) / mat * 100
        print("Maximum pd: %f" % np.nanmax(pd))
        print("Average pd: %f" % np.nanmean(pd))


def rotmat(rot_ax,theta):
    """
    rotmat
    Generate a 3D matrix that rotates a vector around the vector rot_ax by
    angle theta.

    Parameters:
     - rot_ax
     - theta
    """
    rot_ax = rot_ax/math.sqrt(np.dot(rot_ax,rot_ax))
    a = math.cos(theta/2)
    b,c,d = -rot_ax*math.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
