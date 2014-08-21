
"""
.. module:: sw.networks.interface
   :platform: Unix, Windows
   :synopsis: A useful module indeed.

.. moduleauthor:: Bertrand Delforge <b.delforge@sron.nl>

"""

import numpy as np
import sw.geometry as geo
import collections

InterfaceGeometry = collections.namedtuple("InterfaceGeometry",
                                           "theta_a theta_b "
                                           "area_compensation_a "
                                           "area_compensation_b "
                                           "P S "
                                           "R12 R13 "
                                           "R21 R24 "
                                           "R31 R34 "
                                           "R42 R43 "
                                           "k2 k3 k4")

def InterfaceNormal(na, nb):
    """
    na: refractive index on the side of the port 0.
    nb: refractive index on the side of the port 1.

    The result has identical components in the three dimensions.
    This is fine as long as the incoming wave is transverse.

    """
    ra, ta = geo.FresnelNormal(na, nb)
    rb, tb = geo.FresnelNormal(nb, na)
    I = np.identity(3, dtype=complex)
    S00 = ra * I
    S01 = tb * I
    S10 = ta * I
    S11 = rb * I
    return np.array(np.bmat([[S00, S01],
                             [S10, S11]]))

def _InterfaceGeometry(na, nb, n, k1):
    """
    na: refractive index on the side of the ports 0 and 1 (A side).
    nb: refractive index on the side of the ports 2 and 3 (B side).
    n : normal to the surface, point to the A side.
    k1: direction of incidence for the port 0.
    """
    u = geo.ComputeIncidencePlaneNormal(n, k1)
    # Because of the way we made u, theta_a is always going to be positive.
    # So instead of this:
    #     theta_a = geo.ComputeSignedAngleBetween(u, n, k1)
    # We can just use the sign-less version.
    theta_a = geo.ComputeAngleBetween(n, k1)
    if not (0 <= theta_a % geo.TAU <= geo.TAU / 4):
        raise ValueError(u"Wrong side of the surface.")

    theta_b = geo.Snell(na, nb, theta_a)
    # Incident field decomposition.
    P, S = geo.MakeParaPerpDecompositionMatrices(u)

    # The steeper the angle, the smaller the cross section of a finite-size
    # beam. Going from low n to high n increases the angle. It also yields t >
    # 1. Energy is conserved because there is more field in a smaller region.
    # With infinite plane wave, it's a matter of density. Anyway, this is why we
    # have these parameters that compensate for the change of area after a
    # transmission.
    area_compensation_b = np.sqrt(np.cos(theta_a) / np.cos(theta_b))
    area_compensation_a = np.sqrt(np.cos(theta_b) / np.cos(theta_a))
    a21 = -2 * theta_a
    a31 = theta_b - theta_a
    a41 = -theta_a - theta_b
    a43 = -2 * theta_b
    R12 = geo.RotationAroundAxisMatrix(u, -a21)
    R13 = geo.RotationAroundAxisMatrix(u, -a31)
    R21 = geo.RotationAroundAxisMatrix(u, a21)
    R31 = geo.RotationAroundAxisMatrix(u, a31)
    R34 = geo.RotationAroundAxisMatrix(u, -a43)
    R41 = geo.RotationAroundAxisMatrix(u, a41)
    R43 = geo.RotationAroundAxisMatrix(u, a43)
    R24 = R31
    R42 = R13
    k2 = -R21.dot(k1)
    k3 = R31.dot(k1)
    k4 = R41.dot(k1)
    return InterfaceGeometry(theta_a=theta_a, theta_b=theta_b,
                             area_compensation_a=area_compensation_a,
                             area_compensation_b=area_compensation_b,
                             P=P, S=S,
                             R12=R12, R13=R13, R21=R21, R24=R24,
                             R31=R31, R34=R34, R42=R42, R43=R43,
                             k2=k2, k3=k3, k4=k4)
#     return ((theta_a, theta_b),
#             (area_compensation_a, area_compensation_b),
#             (P, S),
#             (R12, R13, R21, R31, R34, R43, R24, R42),
#             (k2, k3, k4))

def InterfaceOblique(na, nb, n, k1):
    ig = _InterfaceGeometry(na, nb, n, k1)
#     theta_a, theta_b = intgeo.theta_a, intgeo.theta_b
#     area_compensation_a, area_compensation_b = intgeo[1]
#     P, S = intgeo[2]
#     R12, R13, R21, R31, R34, R43, R24, R42 = intgeo[3]
#     k2, k3, k4 = intgeo[4]
    #
    rpa, tpa, rsa, tsa = geo.FresnelOblique(na, nb, ig.theta_a)
    rpb, tpb, rsb, tsb = geo.FresnelOblique(nb, na, ig.theta_b)
    tpa *= ig.area_compensation_a
    tsa *= ig.area_compensation_a
    tpb *= ig.area_compensation_b
    tsb *= ig.area_compensation_b
    ra = rpa * ig.P + rsa * ig.S
    rb = rpb * ig.P + rsb * ig.S
    ta = tpa * ig.P + tsa * ig.S
    tb = tpb * ig.P + tsb * ig.S
    S12 = ig.R12.dot(ra)
    S13 = ig.R13.dot(tb)
    S21 = ig.R21.dot(ra)
    S24 = ig.R24.dot(tb)
    S31 = ig.R31.dot(ta)
    S34 = ig.R34.dot(rb)
    S42 = ig.R42.dot(ta)
    S43 = ig.R43.dot(rb)
    ZZZ = np.zeros((3, 3), dtype=complex)
    S = np.array(np.bmat([[ZZZ, S12, S13, ZZZ],
                          [S21, ZZZ, ZZZ, S24],
                          [S31, ZZZ, ZZZ, S34],
                          [ZZZ, S42, S43, ZZZ]]))

    return S, ig.k2, ig.k3, ig.k4
