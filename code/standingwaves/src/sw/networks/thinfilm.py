import numpy as np
import sw.geometry as geo
from distance import ComputeSpaceGain
import collections

ThinFilmGeometry = collections.namedtuple("ThinFilmGeometry",
                                           "theta_a "
                                           "P S "
                                           "R12 R21 R34 R43 "
                                           "k2 k3 k4")

def _ThinFilmRT(n1, n2, thickness, frequency, angle1):
    angle2 = geo.Snell(n1, n2, angle1)  # Angle inside the film.
    rp12, tp12, rs12, ts12 = geo.FresnelOblique(n1, n2, angle1)
    rp21, tp21, rs21, ts21 = geo.FresnelOblique(n2, n1, angle2)
    path_length = thickness / np.cos(angle2)
    a = ComputeSpaceGain(n2, path_length, frequency)
    # A one way transmission has a gain of t12 * a * t21.  Then, on top of that,
    # there are the infinite reflections. Each double reflection brings an
    # additional gain of (rp21 * a) ** 2, that I name q.  The sum of an infinite
    # series of ratio q is 1 / (1-q).  This modulates the original one way
    # transmission.
    tp = tp12 * tp21 * a / (1 - (rp21 * a) ** 2)
    ts = ts12 * ts21 * a / (1 - (rs21 * a) ** 2)
    # The reflection starts a bit differently, but then, the same round trips
    # happen and we can reuse tp and ts.  Indeed, it's as if, instead of
    # reflecting from one port, we were transmitting from the other.
    rp = rp12 + rp21 * a * tp
    rs = rs12 + rs21 * a * ts
    # Since the thin film has a thickness, the volume it occupies is unavailable
    # for anything else.  Therefore we must remove some n1-space.  This path
    # length corresponds to the thickness of the film seen along the incident
    # direction of propagation.  The effect is the same in transmission and
    # reflection.  It also works the same way on each side of the film.
    path_length = thickness / np.cos(angle1)
    space_neg = ComputeSpaceGain(n1, -path_length, frequency)
    rp *= space_neg
    rs *= space_neg
    tp *= space_neg
    ts *= space_neg
    return rp, tp, rs, ts

def _ThinFilmGeometry(n, k1):
    u = geo.ComputeIncidencePlaneNormal(n, k1)
    # Because of the way we made u, theta_a is always going to be positive.
    # So instead of this:
    #     theta_a = geo.ComputeSignedAngleBetween(u, n, k1)
    # We can just use the sign-less version.
    theta_a = geo.ComputeAngleBetween(n, k1)
    if not (0 <= theta_a % geo.TAU <= geo.TAU / 4):
        raise ValueError(u"Wrong side of the surface.")
    # Incident field decomposition.
    P, S = geo.MakeParaPerpDecompositionMatrices(u)
    # Propagation and field rotations.
    # Ports 1 and 3 have the same direction,
    # ports 2 and 4 have the same direction.
    # Going from 1/3 to 2/4 removes 2*theta_a
    a = 2 * theta_a
    Rpos = geo.RotationAroundAxisMatrix(u, a)
    Rneg = geo.RotationAroundAxisMatrix(u, -a)
    R12 = Rpos
    R21 = Rneg
    R34 = Rpos
    R41 = Rneg
    R43 = Rneg
    k2 = -R21.dot(k1)
    k3 = k1
    k4 = R41.dot(k1)
    return ThinFilmGeometry(theta_a=theta_a,
                            P=P, S=S,
                            R12=R12, R21=R21, R34=R34, R43=R43,
                            k2=k2, k3=k3, k4=k4)

def ThinFilmOblique(n1, n2, thickness, frequency, tb, att, k1):
    """Thin film of material n2 at non-normal incidence.  4 ports.

    """
    n = tb.rot(*att).dot(tb.rest)
    tg = _ThinFilmGeometry(n, k1)
    rp, tp, rs, ts = _ThinFilmRT(n1, n2, thickness, frequency, tg.theta_a)
    # Reflections: rotations.
    Sr = rp * tg.P + rs * tg.S
    S12 = tg.R12.dot(Sr)
    S21 = tg.R21.dot(Sr)
    S34 = tg.R34.dot(Sr)
    S43 = tg.R43.dot(Sr)
    # Transmission: no rotation.
    S13 = S24 = S31 = S42 = tp * tg.P + ts * tg.S
    # Impossible paths.
    ZZZ = np.zeros((3, 3), dtype=complex)
    S = np.array(np.bmat([[ZZZ, S12, S13, ZZZ],
                          [S21, ZZZ, ZZZ, S24],
                          [S31, ZZZ, ZZZ, S34],
                          [ZZZ, S42, S43, ZZZ]]))
    return S, tg.k2, tg.k3, tg.k4

def ThinFilmNormal(n1, n2, n3, thickness, frequency, tb, attitude):
    """Thin film of material n2 at normal incidence.  2 ports.

    """
    r12, t12 = geo.FresnelNormal(n1, n2)
    r21, t21 = geo.FresnelNormal(n2, n1)
    r32, t32 = geo.FresnelNormal(n3, n2)
    r23, t23 = geo.FresnelNormal(n2, n3)
    a2 = ComputeSpaceGain(n2, thickness, frequency)
    frac = a2 / (1 - a2 * a2 * r21 * r23)
    r13 = r12 + t12 * t21 * a2 * r23 * frac
    r31 = r32 + t32 * t23 * a2 * r21 * frac
    t13 = t12 * t23 * frac
    t31 = t32 * t21 * frac
    # Where the film is, there is no air.
    a1 = ComputeSpaceGain(n1, -thickness / 2, frequency)
    a3 = ComputeSpaceGain(n3, -thickness / 2, frequency)
    r13 = a1 * a1 * r13
    r31 = a3 * a3 * r31
    t13 = a1 * a3 * t13
    t31 = a1 * a3 * t31
    # Jones matrices.
    I = np.identity(3, dtype=complex)  # Last column meaningless for transverse.
    S11 = r13 * I
    S12 = t31 * I
    S21 = t13 * I
    S22 = r31 * I
    # Attitude corrections.
    R = tb.rot(*attitude)
    Rt = R.T
    S11 = R.dot(S11).dot(Rt)
    S12 = R.dot(S12).dot(Rt)
    S21 = R.dot(S21).dot(Rt)
    S22 = R.dot(S22).dot(Rt)
    #
    S = np.array(np.bmat([[S11, S12],
                          [S21, S22]]))
    return S

