'''
Created on Jun 18, 2013

@author: delforge
'''
import numpy as np
import scipy.constants
import gain
TAU = 2 * np.pi

def ComputeSpaceGain(n, length, frequency):
    k0 = TAU * frequency / scipy.constants.speed_of_light  # In vacuum.
    k = k0 * n  # In the medium.  Handles n complex.
    return np.exp(1j * k * length)

def Distance(n, length, frequency):
    return gain.Gain(ComputeSpaceGain(n, length, frequency))

def Distance1(n, length, frequency):
    return gain.Gain1(ComputeSpaceGain(n, length, frequency))