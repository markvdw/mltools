###############################################################################
# linalg.py
# Some useful linear algebra utilities. Maybe in the future a full library to
# do certain common transformations?
#
# Mark van der Wilk (mv310@cam.ac.uk)
###############################################################################

import math

import numpy as np

def rotmat(rot_ax,theta):
    '''
    rotmat
    Generate a 3D matrix that rotates a vector around the vector rot_ax by
    angle theta.
    
    Parameters:
     - rot_ax
     - theta
    '''
    rot_ax = rot_ax/math.sqrt(np.dot(rot_ax,rot_ax))
    a = math.cos(theta/2)
    b,c,d = -rot_ax*math.sin(theta/2)
    return np.array([[a*a+b*b-c*c-d*d, 2*(b*c-a*d), 2*(b*d+a*c)],
                     [2*(b*c+a*d), a*a+c*c-b*b-d*d, 2*(c*d-a*b)],
                     [2*(b*d-a*c), 2*(c*d+a*b), a*a+d*d-b*b-c*c]])
