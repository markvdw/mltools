###############################################################################
# drawtools.py
# Some drawing tools.
###############################################################################

import numpy as np
import numpy.linalg as linalg
import scipy as sp
import matplotlib as mplot
import matplotlib.pyplot as plt

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
