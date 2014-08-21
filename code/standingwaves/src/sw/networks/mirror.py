
import numpy as np
import sw.geometry as geo

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

def AnisotropicMirrorNormal(rx, ry, rot):
    irot = rot.T
    S11 = np.array([[rx, 0, 0],
                    [0, ry, 0],
                    [0, 0, 0]])
    return rot.dot(S11).dot(irot)

def Rooftop(perfection, na, nb, rot):
    if not (0 <= perfection <= 1):
        raise ValueError("the `perfection` parameter must be between 0 and 1")
    p = np.sqrt(perfection)
    i = np.sqrt(1 - perfection)
    ra, _ = geo.FresnelNormal(na, nb)
    # There are two reflections.
    ra2 = ra ** 2
    # The reflection along x does not flip, but that along y does.
    rx = ra * i + ra2 * p
    ry = ra * i - ra2 * p
    return AnisotropicMirrorNormal(rx, ry, rot)

def PolarizationScrambler(r, rot):
    S11 = np.array([[r, r, 0],
                    [r, r, 0],
                    [0, 0, 0]])
    irot = rot.T
    return rot.dot(S11).dot(irot)
