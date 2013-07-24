'''
Created on Jun 28, 2013

@author: delforge
'''
from __future__ import division
import itertools
import unittest
import random
import numpy as np
import geometry as geo

class TestRotXYZ(unittest.TestCase):
    def testZero(self):
        """Rotation by an angle of 0 has no effect."""
        v = np.array([1, 1, 1], dtype=complex)
        vrx = geo.rotX(0).dot(v)
        vry = geo.rotY(0).dot(v)
        vrz = geo.rotZ(0).dot(v)
        self.assertTrue(np.allclose(vrx, v))
        self.assertTrue(np.allclose(vry, v))
        self.assertTrue(np.allclose(vrz, v))
    def testQuarterTurn(self):
        """Rotation by 90 degrees is correct."""
        v = np.array([1, 1, 1], dtype=complex)
        angle = geo.TAU / 4
        vrx = geo.rotX(angle).dot(v)
        vry = geo.rotY(angle).dot(v)
        vrz = geo.rotZ(angle).dot(v)
        vx = np.array([1, -1, 1], dtype=complex)
        vy = np.array([1, 1, -1], dtype=complex)
        vz = np.array([-1, 1, 1], dtype=complex)
        self.assertTrue(np.allclose(vrx, vx))
        self.assertTrue(np.allclose(vry, vy))
        self.assertTrue(np.allclose(vrz, vz))

class TestTaitBryan(unittest.TestCase):
    def testRotinv(self):
        """The inverse of a rotation matrix is its transpose."""
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        for H, P, R in itertools.permutations(range(3)):
            tb = geo.TaitBryan(H, P, R)
            hprs = ((rnd(), rnd(), rnd()) for _ in range(100))
            for h, p, r in hprs:
                R = tb.rot(h * geo.TAU, p * geo.TAU, r * geo.TAU)
                self.assertTrue(np.allclose(R.T, np.linalg.inv(R)))
    def testRest(self):
        """Roll has no effect on the rest vector."""
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        for H, P, R in itertools.permutations(range(3)):
            tb = geo.TaitBryan(H, P, R)
            rs = (rnd() for _ in range(100))
            for r in rs:
                R = tb.rot(0, 0, r * geo.TAU)
                self.assertTrue(np.allclose(R.dot(tb.rest), tb.rest))
    def testH(self):
        """Heading properly refers to x, y or z."""
        v = np.array([1, 1, 1], dtype=complex)
        angle = geo.TAU / 4
        Rx = geo.TaitBryan(0, 1, 2).rot(angle, 0, 0)
        Ry = geo.TaitBryan(1, 2, 0).rot(angle, 0, 0)
        Rz = geo.TaitBryan(2, 0, 1).rot(angle, 0, 0)
        vx = np.array([1, -1, 1], dtype=complex)
        vy = np.array([1, 1, -1], dtype=complex)
        vz = np.array([-1, 1, 1], dtype=complex)
        self.assertTrue(np.allclose(Rx.dot(v), vx))
        self.assertTrue(np.allclose(Ry.dot(v), vy))
        self.assertTrue(np.allclose(Rz.dot(v), vz))
    def testP(self):
        """Pitch properly refers to x, y or z."""
        v = np.array([1, 1, 1], dtype=complex)
        angle = geo.TAU / 4
        Ry = geo.TaitBryan(0, 1, 2).rot(0, angle, 0)
        Rz = geo.TaitBryan(1, 2, 0).rot(0, angle, 0)
        Rx = geo.TaitBryan(2, 0, 1).rot(0, angle, 0)
        vx = np.array([1, -1, 1], dtype=complex)
        vy = np.array([1, 1, -1], dtype=complex)
        vz = np.array([-1, 1, 1], dtype=complex)
        self.assertTrue(np.allclose(Rx.dot(v), vx))
        self.assertTrue(np.allclose(Ry.dot(v), vy))
        self.assertTrue(np.allclose(Rz.dot(v), vz))
    def testR(self):
        """Roll properly refers to x, y or z."""
        v = np.array([1, 1, 1], dtype=complex)
        angle = geo.TAU / 4
        Rz = geo.TaitBryan(0, 1, 2).rot(0, 0, angle)
        Rx = geo.TaitBryan(1, 2, 0).rot(0, 0, angle)
        Ry = geo.TaitBryan(2, 0, 1).rot(0, 0, angle)
        vx = np.array([1, -1, 1], dtype=complex)
        vy = np.array([1, 1, -1], dtype=complex)
        vz = np.array([-1, 1, 1], dtype=complex)
        self.assertTrue(np.allclose(Rx.dot(v), vx))
        self.assertTrue(np.allclose(Ry.dot(v), vy))
        self.assertTrue(np.allclose(Rz.dot(v), vz))
    def testOrder(self):
        """Applies Roll then Pitch then Heading."""
        v = np.array([1, 1, 1], dtype=complex)
        # I am looking for angles that have a rational sine and a rational
        # cosine. The solutions are linked to pythagorean triples a b c such as
        # a2 + b2 = c2.  I can have cos = a/c and sin = b/c, or the other way
        # around. I also want the number of decimals to be relatively low
        # because I am doing some computations by hand on a piece of paper. The
        # first three triples that make me happy are:
        #
        #  3,   4,   5 :  3 /   5 = .6   and   4 /   5 = .8
        #  7,  24,  25 :  7 /  25 = .28  and  24 /  25 = .96
        # 44, 117, 128 : 44 / 125 = .352 and 117 / 128 = .936
        #
        # These values are exact.  All the other triples seem to have an
        # infinity of decimals, or at least way too many.
        h = np.arccos(44 / 125)
        p = np.arccos(7 / 25)
        r = np.arccos(3 / 5)
        R0 = geo.TaitBryan(0, 1, 2).rot(h, p, r)  # Right handed.
        R1 = geo.TaitBryan(0, 2, 1).rot(h, p, r)  # Left handed.
        R2 = geo.TaitBryan(1, 0, 2).rot(h, p, r)  # Left handed.
        R3 = geo.TaitBryan(1, 2, 0).rot(h, p, r)  # Right handed.
        R4 = geo.TaitBryan(2, 0, 1).rot(h, p, r)  # Right handed.
        R5 = geo.TaitBryan(2, 1, 0).rot(h, p, r)  # Left handed.
        vr0 = R0.dot(v)
        vr1 = R1.dot(v)
        vr2 = R2.dot(v)
        vr3 = R3.dot(v)
        vr4 = R4.dot(v)
        vr5 = R5.dot(v)
        # The components of the vectors down there are exact.
        a, b, c = .904, .051008, 1.476544  # Right handed.
        d, e, f = -.568, .758848, 1.449664  # Left handed.
        v0 = np.array([a, b, c], dtype=complex)
        v1 = np.array([d, e, f], dtype=complex)
        v2 = np.array([f, d, e], dtype=complex)
        v3 = np.array([c, a, b], dtype=complex)
        v4 = np.array([b, c, a], dtype=complex)
        v5 = np.array([e, f, d], dtype=complex)
        self.assertTrue(np.allclose(vr0, v0))
        self.assertTrue(np.allclose(vr1, v1))
        self.assertTrue(np.allclose(vr2, v2))
        self.assertTrue(np.allclose(vr3, v3))
        self.assertTrue(np.allclose(vr4, v4))
        self.assertTrue(np.allclose(vr5, v5))
        # It's kinda funny. For a left handed tb, you can map the components d,
        # e and f directly onto the axes of the tb definition: d=0, e=2, f=1.
        # However, for the right handed case it does not work. Indeed, abc=012
        # for v1, abc=201 for v3 and abc=120 for v4.
    def testHp(self):
        """Retrieve the heading and pitch of a vector."""
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        for H, P, R in itertools.permutations(range(3)):
            tb = geo.TaitBryan(H, P, R)
            hprs = ((rnd(), rnd(), rnd()) for _ in range(100))
            for h, p, r in hprs:
                h = (h - .5) * geo.TAU  # Between -tau/2 and tau/2.
                p = (p - .5) * geo.TAU / 2  # Between -tau/4 and tau/4.
                r *= geo.TAU  # Totally free, should not matter.
                vr = tb.rot(h, p, r).dot(tb.rest)
                hr, pr = tb.hp(vr)
                self.assertAlmostEqual(hr, h, 7)
                self.assertAlmostEqual(pr, p, 7)
    def testAround(self):
        r = geo.TAU / 3  # Rotation of that angle
        u = np.array([1, 1, 1])  # around this axis
        v = np.array([1, 0, 0])  # of that vector
        vv = np.array([0, 1, 0])  # gives that vector.
        for H, P, R in itertools.permutations(range(3)):
            tb = geo.TaitBryan(H, P, R)
            h, p = tb.hp(u)
            vr = tb.around(h, p, r).dot(v)
            self.assertTrue(np.allclose(vr, vv))

class TestSnell(unittest.TestCase):
    def testZero(self):
        angle_t = geo.Snell(2, 3, 0)
        self.assertEqual(angle_t, 0)
    def testTotalReflection(self):
        self.assertTrue(np.isnan(geo.Snell(10, 1, geo.TAU / 3)))
    def testStupidAngle(self):
        self.assertTrue(np.isnan(geo.Snell(1, 1, 2 * geo.TAU / 3)))
        self.assertTrue(np.isnan(geo.Snell(1, 1, 2 * -geo.TAU / 3)))
    def testNoChange(self):
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        for a, n in ((rnd(), rnd()) for _ in range(100)):
            actual_n = n * 100 * .00001
            if a == 0:
                continue  # Tangent incidence is evil.
            angle_i = (a - .5) * geo.TAU / 2
            angle_t = geo.Snell(actual_n, actual_n, angle_i)
            self.assertAlmostEqual(angle_t, angle_i)
    def testDecentValues(self):
        angle_i = -geo.TAU / 8
        # These contrived refractive indices will convert sqrt(2)/2 into
        # sqrt(3)/2, transforming 45 degrees (tau/8) into 60 degrees (tau/6).
        ni = 1 / np.sqrt(2)
        nt = 1 / np.sqrt(3)
        angle_t = geo.Snell(ni, nt, angle_i)
        self.assertAlmostEqual(angle_t, -geo.TAU / 6)

class TestFresnelNormal(unittest.TestCase):
    def testNoChange(self):
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        for n in (rnd() for _ in range(100)):
            actual_n = n * 100 * .00001
            r, t = geo.FresnelNormal(actual_n, actual_n)
            self.assertAlmostEqual(r, 0)
            self.assertAlmostEqual(t, 1)
    def testSigns(self):
        r, t = geo.FresnelNormal(1, 2)
        self.assertTrue(r < 0)
        self.assertTrue(t > 0)
        r, t = geo.FresnelNormal(2, 1)
        self.assertTrue(r > 0)
        self.assertTrue(t > 0)
    def testDecentValues(self):
        r, t = geo.FresnelNormal(1, 2)
        self.assertAlmostEqual(r, -1 / 3)
        self.assertAlmostEqual(t, 2 / 3)

class TestFresnelOblique(unittest.TestCase):
    def testBrewsterAngle(self):
        """Angle for which the reflection is totally (s) polarized."""
        ni = 1
        nt = 2
        theta_b = np.arctan(nt / ni)
        rp, tp, rs, ts = geo.FresnelOblique(ni, nt, theta_b)  # @UnusedVariable
        self.assertAlmostEqual(rp, 0)
    def testCriticalAngle(self):
        ni = 2
        nt = 1
        theta_c = np.arcsin(nt / ni)
        # This theta_c equals 30 degrees.  For an incidence theta_i = theta_c,
        # the angle of refraction is 90 degrees, sin(theta_t) = 1.
        rp, tp, rs, ts = geo.FresnelOblique(ni, nt, theta_c)
        self.assertAlmostEqual(rp, -1)
        self.assertAlmostEqual(tp, 4)
        self.assertAlmostEqual(rs, 1)
        self.assertAlmostEqual(ts, 2)
        # Beyond the critical angle theta_c, there is no refraction, everything
        # is reflected.  We do not support this functionality for now.
        above_c = theta_c * 1.0001
        rp, tp, rs, ts = geo.FresnelOblique(ni, nt, above_c)
        self.assertTrue(np.isnan(rp))
        self.assertTrue(np.isnan(tp))
        self.assertTrue(np.isnan(rs))
        self.assertTrue(np.isnan(ts))

class TestComputeIncidencePlaneNormal(unittest.TestCase):
    def testNoIncidencePlane(self):
        n = np.array([0, 0, 1])
        k = np.array([0, 0, 1])
        self.assertRaises(ZeroDivisionError, geo.ComputeIncidencePlaneNormal, n, k)
    def testDecentValues(self):
        n = np.array([0, 0, 1])
        k = np.array([0, 1, 1])  # Note that it's not a unit vector.
        u = np.array([-1, 0, 0])  # Expected result, unit nonetheless.
        ur = geo.ComputeIncidencePlaneNormal(n, k)
        self.assertTrue(np.allclose(ur, u))

class TestQuaternion(unittest.TestCase):
    # I go straight to the useful part: rotations.  If these work, it means that
    # the low level things like multiplication work too.
    def testQuatEulerEquiv(self):
        # We are testing a lot of things at the same time here.
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        # This defines a rotation axis and an rotation angle around that axis.
        tb = geo.TaitBryan(0, 1, 2)
        def randAngle():
            # We don't limit our tests to [0, TAU].
            return (rnd() - .5) * geo.TAU * 10
        hpa_s = ((randAngle(), randAngle(), randAngle()) for _ in range(100))
        # We will apply these rotations to all these vectors:
        def randCoord():
            return (rnd() - .5) * 10
        vs = (np.array([randCoord(), randCoord(), randCoord()]) for _ in range(20))
        for h, p, angle in hpa_s:
            # Pure euler matrix method.
            Re = tb.around(h, p, angle)
            # Various quaternion methods.
            axis = tb.rot(h, p, 0).dot(tb.rest)
            q = geo.RotationAroundAxisQuaternion(axis, angle)
            Rq1 = geo.RotationMatrixFromQuaternion(q)
            Rq2 = geo.RotationAroundAxisMatrix(axis, angle)
            # Compare the matrices from Euler to the matrices from the quaternions.
            self.assertTrue(np.allclose(Re, Rq1))
            self.assertTrue(np.all(Rq1 == Rq2))
            # Now, compare Euler rotation with Quaternion rotation.
            for v in vs:
                vr = Re.dot(v)  # Matrix multiplication.
                vq = geo.QuatRotate(q, v)  # Quaternion Hamilton product.
                self.assertTrue(np.allclose(vr, vq))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
