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

def Snell(ni, nt, anglei):
    # ni si = nt st
    # st = si ni / nt
    # t = arcsin(si ni / nt)
    return np.arcsin(np.sin(anglei) * ni.real / nt.real)

def Fresnel(ni, nt, anglei):
    """
    Returns:
        rp, tp, rs, ts
        rp: parallel reflection
        tp: parallel transmission
        rs: perpendicular reflection
        ts: perpendicular transmission

    The terms "parallel" and "perpendicular" are relative to the plane of
    incidence, defined as the plane that contains the normal to the surface of
    the thin film and the direction of propagation.  For this plane to exist,
    the direction of propagation must not be collinear to the normal to the
    surface, or in other words, the light should not arrive at normal incidence.

    Reference: Echt 2002, chapter 4.

    """
    anglet = Snell(ni, nt, anglei)
    cosi = np.cos(anglei)
    cost = np.cos(anglet)
    den_p = ni * cost + nt * cosi
    den_s = ni * cosi + nt * cost
    # Be super careful here.  The signs depend on some arbitrary choice of
    # directions for the electric and magnetic field.  Echt warns us about that
    # in one extensive paragraph concluded in "To avoid confusion, they [the
    # Fresnel equations] must be related to the specific field directions from
    # which they were derived.".  I do not simply copy paste the equations given
    # by Echt because they are inconsistent.  Indeed, if you take his, then for
    # an angle of 0 rp and rs spit out two opposite results, which makes no
    # sense: it is because the conventions used by Echt don't match for the two
    # cases.  Here, I 'fix' them.  I'm not blaming Echt, because he was damn
    # well aware of it and he made it very clear.  I just wished he had dropped
    # a line on the normal incidence situation.
    rp = (ni * cost - nt * cosi) / den_p
    tp = (2 * ni * cosi) / den_p
    rs = (ni * cosi - nt * cost) / den_s
    ts = (2 * ni * cosi) / den_s
    return rp, tp, rs, ts

def FresnelNormal(ni, nt):
    # The incidence plane is undefined in the case of normal incidence.
    den = ni + nt
    r = (ni - nt) / den
    t = (2 * ni) / den
    return r, t

def ComputeAngleBetween(v1, v2):
    # This uses the dumb fact that V1.V2 = ||V1|| ||V2|| cos(V1, V2).
    cos = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos)

def ComputeIncidencePlaneNormal(surface_normal, propagation_direction):
    """Define the plane of incidence.

    The plane of incidence is the plane that contains both `surface_normal` and
    `propagation_direction`.

    In case of normal incidence, the incidence plane is undefined and we return
    ZeroDivisionError.

    ValueError("Incidence from the wrong side of the surface.") is raised when
    the surface normal and the propagation direction point in opposite ways
    (once projected on each other).  For example, if the surface normal is (0,
    0, 1) and the propagation direction (0, 0, -1), then the wave comes from the
    wrong side of the surface (contrived example using normal incidence, but you
    get the idea, a dot product is present to make it work at oblique
    incidence).

    We do not check that the normal and the propagation direction are unit
    vectors but it does not matter because we normalize the result.

    A plane has two normals.  We return the normal that is pointing toward the
    cross-product of `surface_normal` and `propagation_direction` in that order.

    """
    if surface_normal.dot(propagation_direction) < 0:
        raise ValueError("Incidence from the wrong side of the surface.")
    incidence_normal = np.cross(surface_normal, propagation_direction)
    norm = np.linalg.norm(incidence_normal)
    if norm == 0:
        raise ZeroDivisionError("Normal incidence detected.")
    incidence_normal /= norm
    return incidence_normal

def MakeParaPerpDecompositionMatrices(plane_normal):
    """Useful for Fresnel equations for instance.

    Given a plane defined by its normal vector `plane_normal`, assumed to be a
    unit vector, this function returns two matrices.

    The Perpendicular matrix S projects a vector v on the plane normal. The
    Parallel matrix P returns the difference between the original vector and its
    projection on the normal.  P v + S v = v.

    This function returns P and S, in that order.

    How do I project onto a vector?  I project v onto n and get w.  w and n are
    collinear.  So w equals n times a scalar constant.  That constant is tied to
    the dot product of v and n.  If n is a unit vector (and here it is), then
    that constant IS the dot product.  So we need a matrix that transforms v
    into w = n (n.v).

    n and v are column vectors.  The dot product n.v can be written as the
    matrix product nT x v with nT the transpose of n.  w = n (nT x v).

    Does n (nT x v) = (n x nT) x v ?  Yes it does.  I don't know why I can
    replace an external product by an internal product and get away with it, but
    if I just write down both expressions they give the same result.

    """
    n = plane_normal.reshape(1, 3)  # Switch to 2D otherwise numpy gives scalar result.
    S = n.dot(n.T)
    P = np.identity(3) - S
    return P, S
