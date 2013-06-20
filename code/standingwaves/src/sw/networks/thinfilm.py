import numpy as np
import sw.geometry as geo
from distance import ComputeSpaceGain

def _ThinFilmRT(n1, n2, thickness, frequency, angle1):
    angle2 = geo.Snell(n1, n2, angle1)  # Angle inside the film.
    rp12, tp12, rs12, ts12 = geo.Fresnel(n1, n2, angle1)
    rp21, tp21, rs21, ts21 = geo.Fresnel(n2, n1, angle2)
    path_length = thickness / np.cos(angle2)
    a = ComputeSpaceGain(n2, path_length, frequency)
    # A one way transmission has a gain of t12 * a * t21.  Then, on top of that,
    # there are the infinite reflections. Each double reflection brings an
    # additional gain of (rp21 * a) ** 2, that I name q.  The sum of an infinite
    # series of ratio q is 1 / (1-q).  This modulates the original one way
    # transmission.
    tp = tp12 * tp21 * a / (1 - (rp21 * a) ** 2)
    ts = ts12 * ts21 * a / (1 - (rp21 * a) ** 2)
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

def _ThinFilmGeometry(tb, attitude, k1):
    rdir, _ = tb.rot(*attitude)
    surface_normal = rdir.dot(tb.rest)
    angle1 = geo.ComputeAngleBetween(surface_normal, k1)
    assert 0 <= angle1 % geo.TAU <= geo.TAU / 4, "Wrong side of the surface."
    # Incident field decomposition.
    ipn = geo.ComputeIncidencePlaneNormal(surface_normal, k1)
    P, S = geo.MakeParaPerpDecompositionMatrices(ipn)
    # Propagation and field rotations.
    ipn_h, ipn_p = tb.hp(ipn)
    roll21 = -2 * angle1
    roll41 = -2 * angle1
    roll43 = -2 * angle1
    R12, _ = tb.around(ipn_h, ipn_p, -roll21)
    R21, _ = tb.around(ipn_h, ipn_p, roll21)
    R34, _ = tb.around(ipn_h, ipn_p, -roll43)
    R43, _ = tb.around(ipn_h, ipn_p, roll43)
    R41, _ = tb.around(ipn_h, ipn_p, roll41)
    k2 = -R21.dot(k1)
    k3 = k1
    k4 = R41.dot(k1)
    return angle1, P, S, R12, R21, R34, R43, k2, k3, k4

def ThinFilm(n1, n2, thickness, frequency, tb, attitude, k1):
    angle1, P, S, R12, R21, R34, R43, k2, k3, k4 = _ThinFilmGeometry(tb, attitude, k1)
    rp, tp, rs, ts = _ThinFilmRT(n1, n2, thickness, frequency, angle1)
    # Transmission: no rotation.
    S13 = S24 = S31 = S42 = tp * P + ts * S
    # Reflections: rotations.
    Sr = rp * P + rs * S
    S12 = R12.dot(Sr)
    S21 = R21.dot(Sr)
    S34 = R34.dot(Sr)
    S43 = R43.dot(Sr)
    # Impossible paths.
    S11 = S14 = S22 = S23 = S32 = S33 = S41 = S44 = np.zeros((3, 3))
    S = np.array(np.bmat([[S11, S12, S13, S14],
                          [S21, S22, S23, S24],
                          [S31, S32, S33, S34],
                          [S41, S42, S43, S44]]))
    return S, k2, k3, k4
