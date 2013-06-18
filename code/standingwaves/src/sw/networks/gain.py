'''
Created on Jun 18, 2013

@author: delforge
'''
import numpy as np

def Gain(g):
    return np.array([[0, 0, 0, g, 0, 0],
                     [0, 0, 0, 0, g, 0],
                     [0, 0, 0, 0, 0, g],
                     [g, 0, 0, 0, 0, 0],
                     [0, g, 0, 0, 0, 0],
                     [0, 0, g, 0, 0, 0]], dtype=complex)

def Gain1(g):
    return np.array([[0, g], [g, 0]], dtype=complex)
