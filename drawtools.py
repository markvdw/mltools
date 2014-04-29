###############################################################################
# drawtools.py
# Some drawing tools.
###############################################################################

import sys

import numpy as np
import numpy.linalg as linalg

import scipy as sp
import scipy.constants

import matplotlib as mplot
import matplotlib.pyplot as plt
import matplotlib.animation as anim
from mpl_toolkits.mplot3d import Axes3D


def anim_3dfunc(X0, X1, Y, interval=50):
    """
    anim_3dfunc
    Creates an animated plot where points are plotted. Each *column* is a
    different time instance.
    """
    def get_num(V, num):
        if V.ndim == 1:
            v = V
        else:
            v = V[:, num]
        return v

    def update_histu(num, X0, X1, Y, line):
        x0 = get_num(X0, num)
        x1 = get_num(X1, num)
        y = get_num(Y, num)

        line[0].set_data(x0, x1)
        line[0].set_3d_properties(y)
        sys.stdout.write('%i   \r' % num)
        return line

    def clear_histU():
        ls[0].set_data([], [])
        ls[0].set_3d_properties([])
        return ls

    # def clear_histU():
    #     pass

    maxnum = 1
    for d in (X0, X1, Y):
        try:
            maxnum = max(maxnum, d.shape[1])
        except:
            pass
    print maxnum

    figU = plt.figure()
    ax = figU.add_subplot(111, projection='3d')
    ls = ax.plot(get_num(X0, 0), get_num(X1, 0), get_num(Y, 0), 'x')
    histU_anim = anim.FuncAnimation(figU, update_histu, maxnum, fargs=(X0, X1, Y, ls), interval=interval, blit=False, init_func=clear_histU)
    plt.show()

    return ax


def line_sets_2d(set1, set2):
    '''
    line_sets_2d
    Draw lines between points in two sets (2D).
    '''
    plt.plot(np.vstack((set1.T[0, :], set2.T[0, :])), np.vstack((set1.T[1, :], set2.T[1, :])))


def line_sets_3d(ax, set1, set2, opt='r'):
    '''
    line_sets_3d
    Draw lines between points in two sets (3D).
    '''
    # ax.plot3D(np.vstack((set1.T[0, :], set2.T[0, :])),
    #           np.vstack((set1.T[1, :], set2.T[1, :])),
    #           np.vstack((set1.T[2, :], set2.T[2, :])), 'x')
    for s1, s2 in zip(set1, set2):
        plt.plot([s1[0], s2[0]],
                 [s1[1], s2[1]],
                 [s1[2], s2[2]], opt)


def auto_axes_robust(ax, datax, datay, prop=0.95, verbose=False):
    '''
    auto_axes_robust
    Automatically adjust the axes of a plot, but be robust to outliers. Make
    sure that at most the proportion of the data given by 'prop' is actually
    displayed.
    '''
    def _find_robust_range(data, prop=0.95):
        '''
        _find_robust_range
        Required function by auto_axes_robust.
        '''
        numpoints = np.prod(data.shape)
        threshold = prop * numpoints
        r = max([np.abs(np.max(data)), np.abs(np.min(data))])

        while(np.sum(np.abs(data) < r) > threshold):
            r *= 0.9

        return r, np.sum(np.abs(data) < r) / numpoints

    rx, fx = _find_robust_range(datax, prop)
    ry, fy = _find_robust_range(datay, prop)
    ax.set_xlim(-rx, rx)
    ax.set_ylim(-ry, ry)

    if (verbose):
        print('At the least %f is displayed.' % (fx * fy))


def GenEllipsoid(A, c=[0,0,0], divs=100):
    """
    Calculate the mesh for an ellipse. Points satisfy (x-c)'A(x-c) = 1. In
    other words, A is the *precision* matrix when plotting a Gaussian.

    Args:
        A: 3x3 matrix describing the ellipse.
        c: Centre of the ellipse.

    Returns:
        (x, y, z): Tuple of 2x2 grids containing the x, y and z coordinates for
                   the points on the mesh.
    """
    # find the rotation matrix and radii of the axes
    _, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)

    # now carry on with EOL's answer
    u = np.linspace(0.0, 2.0 * np.pi, divs)
    v = np.linspace(0.0, np.pi, divs)
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + c

    return (x, y, z)


def DrawCovarianceContour (ax, A, c=[0,0]):
    """
    Draw an ellipse for a Gaussian with *precision* matrix A.

    Args:
        A: 2x2 matrix describing the ellipse.
        c: Centre of the ellipse.
    """
    lab, l = linalg.eig(linalg.inv(A))

    ell = mplot.patches.Ellipse(xy=c,
                                width=np.sqrt(lab[0])*4,
                                height=np.sqrt(lab[1])*4,
                                angle=-np.arctan2(l[0,1], l[0,0]) / sp.constants.pi * 180)

    ell.set_facecolor([1, 1, 1])

    ax.add_artist(ell)


if __name__ == "__main__":
    import numpy.random as rnd

    A = rnd.randn(2, 2)
    A = A.dot(A.T)

    d = rnd.multivariate_normal(np.zeros(2), A, 1000)

    fig = plt.figure(1)
    plt.clf()
    plt.plot(d[:, 0], d[:, 1], 'x')
    ax = plt.gca()
    DrawCovarianceContour(ax, linalg.inv(A))

    A = np.array([[1,0,0],[0,8,0],[0,0,1]])
    center = [0,0,0]
    x, y, z = GenEllipsoid(A, center)

    plt.show()
    plt.ion()
    fig = plt.figure(2)
    ax = fig.add_subplot(111, projection='3d')
    wframe = ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color='b', alpha=0.2)
    ax.axis([-5, 5, -5, 5])
    ax.set_zlim3d(-5, 5)

    A = np.array([[1., 0., 0.],[0., 1., 0.], [0., 0., 1.]])
    for _ in range(50):
        if wframe is not None:
            ax.collections.remove(wframe)

        x, y, z = GenEllipsoid(A, divs=20)
        wframe = ax.plot_wireframe(x, y, z)
        ax.axis([-2, 2, -2, 2])
        ax.set_zlim3d(-2, 2)
        plt.draw()
        A[1, 1] = A[1,1] * 0.9
