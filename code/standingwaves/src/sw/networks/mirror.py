'''
Created on Jun 18, 2013

@author: delforge
'''
import numpy as np

def Mirror(r):
    return np.array([[r, 0, 0],
                     [0, r, 0],
                     [0, 0, r]], dtype=complex)

def SemiTransparentMirror(r, t):
    return np.array([[r, 0, 0, t, 0, 0],
                     [0, r, 0, 0, t, 0],
                     [0, 0, r, 0, 0, t],
                     [t, 0, 0, r, 0, 0],
                     [0, t, 0, 0, r, 0],
                     [0, 0, t, 0, 0, r]], dtype=complex)
