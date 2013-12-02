
import numpy as np

def MirrorNormal(r):
    """

    """
    return np.array([[r, 0, 0],
                     [0, r, 0],
                     [0, 0, r]], dtype=complex)

def MirrorNormal1(r):
    return np.array([[r]], dtype=complex)

def SemiTransparentMirrorNormal(r, t):
    return np.array([[r, 0, 0, t, 0, 0],
                     [0, r, 0, 0, t, 0],
                     [0, 0, r, 0, 0, t],
                     [t, 0, 0, r, 0, 0],
                     [0, t, 0, 0, r, 0],
                     [0, 0, t, 0, 0, r]], dtype=complex)

def SemiTransparentMirrorNormal1(r, t):
    return np.array([[r, t], [t, r]], dtype=complex)

def TwoWayMirrorNormal(r11, r22, t12, t21, tb, attitude):
    Rd = tb.rot(*attitude)
    Ri = Rd.T
    S11 = np.array([[r11, 0, 0],
                    [0, r11, 0],
                    [0, 0, 0]])
    S12 = np.array([[t12, 0, 0],
                    [0, t12, 0],
                    [0, 0, 0]])
    S21 = np.array([[t21, 0, 0],
                    [0, t21, 0],
                    [0, 0, 0]])
    S22 = np.array([[r22, 0, 0],
                    [0, r22, 0],
                    [0, 0, 0]])
    S11 = Rd.dot(S11).dot(Ri)
    S12 = Rd.dot(S12).dot(Ri)
    S21 = Rd.dot(S21).dot(Ri)
    S22 = Rd.dot(S22).dot(Ri)
    return np.array(np.bmat([[S11, S12], [S21, S22]]))