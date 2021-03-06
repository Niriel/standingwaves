from __future__ import division
import warnings
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
        signature = H - P
        self.right_handed = signature == -1 or signature == 2
    def rot(self, h, p, r):
        """Rotation matrix for the given heading, pitch and roll.

        This returns the direct rotation matrix.  For its inverse, just
        transpose it. Indeed, the inverse of a rotation matrix is its transpose.

        """
        return self.rotH(h).dot(self.rotP(p)).dot(self.rotR(r))
    def hp(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return 0.0, 0.0
        vn = v / norm
        h = np.arctan2(-vn[self.P], vn[self.R])
        p = np.arcsin(vn[self.H])
        if not self.right_handed:
            h = -h
            p = -p
        return h, p
    def around(self, h, p, r):
        """Rotation matrix for an angle r around an axis of given heading and pitch."""
        # First, rotate the universe so that the given axis coincides with the rest axis.
        # Do the rotation by r.
        # Put the universe back in place.
        universe = self.rotH(h).dot(self.rotP(p))
        return universe.dot(self.rotR(r)).dot(universe.T)

def Snell(ni, nt, anglei):
    # Also works for negative angles.
    # ni si = nt st
    # st = si ni / nt
    # t = arcsin(si ni / nt)
    if TAU / 4 < anglei % TAU < 3 * TAU / 4:
        warnings.warn(u"Snell: Angle-of-incidence greater than pi/2.")
        return np.nan
    # The following arcsin may return nan if its argument is out of -1..1. It
    # will also raise a warning.  This is why I do the same with the check
    # regarding the limits of anglei.  I can understand why they return nan
    # rather than crash: they're trying to have total functions.  I may consider
    # trigger a crash if a flag is raised to help with debugging, but by default
    # I stick to numpy's behavior.
    sin = np.sin(anglei) * ni.real / nt.real
    if sin < -1 or sin > 1:
        crit = np.arcsin(nt / ni)
        msg = u"Snell: angle=%f\u00B0 > critical=%f\u00B0" % (np.degrees(anglei), np.degrees(crit))
        warnings.warn(msg)
        print msg
    return np.arcsin(sin)

def FresnelOblique(ni, nt, anglei):
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
    if np.isnan(anglet):
        # Frankly, getting it to work may just be a matter of working with
        # complex values instead of complaining with nans.  But that's not
        # needed for now.
        warnings.warn("Fresnel: Total internal reflection.")
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
    """Fresnel equations for normal incidence.

    This is not really needed since the Fresnel function can do this. However,
    its usage is more clear in the case of normal incidence. Indeed, it does not
    require to specify an angle, and it returns only two values instead of four.

    """
    # The incidence plane is undefined in the case of normal incidence.
    den = ni + nt
    r = (ni - nt) / den
    t = (2 * ni) / den
    # Note: r**2 + t**2 is not 1.  That does not mean that energy is not
    # conserved.  It is.  Thing is, with a change of refractive index
    # comes a change in impedance.  Energy is ExB, and if something is
    # missing in E then it gets in B.  Fresnel here deals with E only.
    # Keep the impedance in mind when computing power.
    return r, t

def ComputeAngleBetween(v1, v2):
    # This uses the dumb fact that V1.V2 = ||V1|| ||V2|| cos(V1, V2).
    cos = v1.dot(v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return np.arccos(cos)

def ComputeSignedAngleBetween(u, r, v):
    """Return the angle around u to go from a reference r to a vector v.

    Returns an angle in ]-pi/2 , pi/2].

    Determining the sign of a rotation angle requires a definition for that
    sign. This is the role of the vector u.  u is the rotation axis, and it
    carries with it the orientation of the rotation.  Positive angles correspond
    to a direct (counter clockwise) rotation, negative angles to a retrograde
    (clockwise) rotation.

    The order of r and v matters.  r is the reference.  The angle starts there.
    v is the destination, the angle ends there.  Swap these two vectors and the
    sign of the result switches.

    """
    a = ComputeAngleBetween(r, v)
    n = np.cross(r, v)
    if n.dot(u) < 0:
        # Strictly negative so that I do not flip the sign of 180 degrees.
        return -a
    return a

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
    n = plane_normal.reshape(3, 1)  # Switch to 2D column otherwise numpy gives scalar result.
    S = n.dot(n.T)
    P = np.identity(3) - S
    return P, S

def RotationAroundAxisQuaternion(axis, angle):
    # Source: wikipedia.  Need better source.
    # I can probably go back to Hamilton's book, if I can find it.
    x, y, z = axis
    c = np.cos(.5 * angle)
    s = np.sin(.5 * angle)
    return np.array([c, x * s, y * s, z * s])

def RotationMatrixFromQuaternion(quat):
    # Source: wikipedia.  Need better source.
    w, x, y, z = quat
    x2 = x * x
    y2 = y * y
    z2 = z * z
    return np.array([[1 - 2 * y2 - 2 * z2,
                      2 * x * y - 2 * z * w,
                      2 * x * z + 2 * y * w],
                     [2 * x * y + 2 * z * w,
                      1 - 2 * x2 - 2 * z2,
                      2 * y * z - 2 * x * w],
                     [2 * x * z - 2 * y * w,
                      2 * y * z + 2 * x * w,
                      1 - 2 * x2 - 2 * y2]])

def RotationAroundAxisMatrix(axis, angle):
    return RotationMatrixFromQuaternion(RotationAroundAxisQuaternion(axis, angle))

def QuatMult(p, q):
    # While waiting to find Hamilton's book:
    # @article{vicci2001quaternions,
    #  title={Quaternions and rotations in 3-space: The algebra and its geometric interpretation},
    #  author={Vicci, Leandra},
    #  journal={Microelectronic Systems Laboratory, Departement of Computer Science, University of North Carolina at Chapel Hill},
    #  year={2001},
    #  publisher={Citeseer}
    # }
    p1, p2, p3, p4 = p
    q1, q2, q3, q4 = q
    return np.array([p1 * q1 - p2 * q2 - p3 * q3 - p4 * q4,
                     p1 * q2 + p2 * q1 + p3 * q4 - p4 * q3,
                     p1 * q3 + p3 * q1 + p4 * q2 - p2 * q4,
                     p1 * q4 + p4 * q1 + p2 * q3 - p3 * q2])

def QuatConj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])

def QuatHamilton(p, q):
    """Return qpq*, quaternion Hamilton product."""
    return QuatMult(q, QuatMult(p, QuatConj(q)))

def QuatRotate(q, v):
    p = np.array([0, v[0], v[1], v[2]])
    p1 = QuatHamilton(p, q)
    return p1[1:]
