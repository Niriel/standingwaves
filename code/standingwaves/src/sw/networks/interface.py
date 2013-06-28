"""
.. module:: sw.networks.interface
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Bertrand Delforge <b.delforge@sron.nl>

"""

import numpy as np
import sw.geometry as geo

def InterfaceNormal(na, nb):
    ra, ta = geo.FresnelNormal(na, nb)
    rb, tb = geo.FresnelNormal(nb, na)
    I = np.identity(3, dtype=complex)
    return np.array(np.bmat([[ra * I, tb * I],
                             [ta * I, rb * I]]))

def InterfaceOblique(na, nb, n, k1):
    theta_a = geo.ComputeAngleBetween(n, k1)
    theta_b = geo.Snell(na, nb, theta_a)
    u = geo.ComputeIncidencePlaneNormal(n, k1)
    a12 = 2 * theta_a - geo.TAU / 2
    a13 = theta_a - theta_b
    a21 = -a12
    a31 = -a13
    a34 = 2 * theta_b - geo.TAU / 2
    a41 = -theta_a - theta_b
    a43 = -a34
    R12 = geo.RotationAroundAxisMatrix(u, a12)
    R13 = geo.RotationAroundAxisMatrix(u, a13)
    R21 = geo.RotationAroundAxisMatrix(u, a21)
    R31 = geo.RotationAroundAxisMatrix(u, a31)
    R34 = geo.RotationAroundAxisMatrix(u, a34)
    R41 = geo.RotationAroundAxisMatrix(u, a41)
    R43 = geo.RotationAroundAxisMatrix(u, a43)
    R24 = R31
    R42 = R13
    P, S = geo.MakeParaPerpDecompositionMatrices(u)
    rpa, tpa, rsa, tsa = geo.FresnelOblique(na, nb, theta_a)
    rpb, tpb, rsb, tsb = geo.FresnelOblique(nb, na, theta_b)
    ra = rpa * P + rsa * S
    rb = rpb * P + rsb * S
    ta = tpa * P + tsa * S
    tb = tpb * P + tsb * S
    S12 = R12.dot(ra)
    S13 = R13.dot(tb)
    S21 = R21.dot(ra)
    S24 = R24.dot(tb)
    S31 = R31.dot(ta)
    S34 = R34.dot(rb)
    S42 = R42.dot(ta)
    S43 = R43.dot(rb)
    ZZZ = np.zeros((3, 3), dtype=complex)
    S = np.array(np.bmat([[ZZZ, S12, S13, ZZZ],
                          [S21, ZZZ, ZZZ, S24],
                          [S31, ZZZ, ZZZ, S34],
                          [ZZZ, S42, S43, ZZZ]]))
    k2 = R21.dot(k1)
    k3 = R31.dot(k1)
    k4 = R41.dot(k1)
    return S, k2, k3, k4
