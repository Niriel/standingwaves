import numpy as np

TAU = 2 * np.pi

def rotX(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[1, 0, 0],
                     [0, c, -s],
                     [0, s, c]])

def rotY(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, 0, s],
                     [0, 1, 0],
                     [-s, 0, c]])

def rotZ(angle):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([[c, -s, 0],
                     [s, c, 0],
                     [0, 0, 1]])

class TaitBryan(object):
    def __init__(self, H, P, R):
        object.__init__(self)
        assert 0 <= H <= 2, "Heading must be in [0, 1, 2]."
        assert 0 <= P <= 2, "Pitch must be in [0, 1, 2]."
        assert 0 <= R <= 2, "Roll must be in [0, 1, 2]."
        assert H != P, "Heading and pitch must differ."
        assert H != R, "Heading and roll must differ."
        assert P != R, "Pitch and roll must differ."
        self.H = H
        self.P = P
        self.R = R
        self.rest = np.zeros(3, float)
        self.rest[R] = 1
        ROTS = [rotX, rotY, rotZ]
        self.rotH = ROTS[H]
        self.rotP = ROTS[P]
        self.rotR = ROTS[R]
    def rot(self, h, p, r):
        rdir = self.rotH(h).dot(self.rotP(p)).dot(self.rotR(r))
        rinv = self.rotR(-r).dot(self.rotP(-p)).dot(self.rotH(-h))
        return rdir, rinv
    def hp(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return 0.0, 0.0
        vn = v / norm
        h = np.arcsin(vn[self.H])
        p = np.arctan2(-vn[self.P], vn[self.R])
        return h, p
    def around(self, h, p, r):
        hp, hm = self.rotH(h), self.rotH(-h)
        pp, pm = self.rotP(p), self.rotP(-p)
        rp, rm = self.rotR(r), self.rotR(-r)
        rdir = hp.dot(pp).dot(rp).dot(pm).dot(hm)
        rinv = hm.dot(pm).dot(rm).dot(pp).dot(hp)
        return rdir, rinv

def Snell(n1, n2, angle1):
    if angle1 == 0:
        return 0.
    sin_angle2 = n1.real / n2.real * np.sin(angle1)
    return np.arcsin(sin_angle2)

def Fresnel(n1, n2, angle1):
    """
    Returns:
        rp, tp, rs, ts
        rp: parallel reflection
        tp: parallel transmission
        rs: perpendicular reflection
        ts: perpendicular transmission

    """
    # TODO: reference to Hecht (1987) and Hecht (2003).
    angle2 = Snell(n1, n2, angle1)
    cos1 = np.cos(angle1)
    cos2 = np.cos(angle2)
    den_p = n2 * cos1 + n1 * cos2
    den_s = n1 * cos1 + n2 * cos2
    rp = (n2 * cos1 - n1 * cos2) / den_p
    tp = (2 * n1 * cos1) / den_p
    rs = (n1 * cos1 - n2 * cos2) / den_s
    ts = (2 * n1 * cos1) / den_s
    return rp, tp, rs, ts

def ComputeAngleBetween(v1, v2):
    # This uses the dumb fact that V1.V2 = ||V1|| ||V2|| cos(V1, V2).
    cos = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos)

def ComputeIncidencePlaneNormal(surface_normal, propagation_direction):
    incidence_normal = np.cross(surface_normal, propagation_direction)
    norm = np.linalg.norm(incidence_normal)
    if norm == 0:
        raise ZeroDivisionError("Normal incidence detected.")
    incidence_normal /= norm
    return incidence_normal

def MakeParaPerpDecompositionMatrices(v):
    # v is the normal to a plane.
    v1 = v.reshape(1, 3)  # Switch to 2D otherwise scalar result.
    S = v1.T.dot(v1)
    P = np.identity(3) - S
    return P, S
