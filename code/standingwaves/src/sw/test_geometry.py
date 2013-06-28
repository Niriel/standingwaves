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
    def testRotRotinv(self):
        """A rotation and its inverse cancel each other.

        This is a regression test using a bunch of random angles for every
        possible Tait-Bryan convention.

        The goal is to make sure that the two rotation matrices returned by
        TaitBryan.rot are inverse of each other.

        """
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        I = np.identity(3)
        for H, P, R in itertools.permutations(range(3)):
            tb = geo.TaitBryan(H, P, R)
            hprs = ((rnd(), rnd(), rnd()) for _ in range(100))
            for h, p, r in hprs:
                Rd, Ri = tb.rot(h * geo.TAU, p * geo.TAU, r * geo.TAU)
                self.assertTrue(np.allclose(Rd.dot(Ri), I))
                self.assertTrue(np.allclose(Ri.dot(Rd), I))
    def testRest(self):
        """Roll has no effect on the rest vector."""
        randomizer = random.Random()
        randomizer.seed(0)
        rnd = randomizer.random
        for H, P, R in itertools.permutations(range(3)):
            tb = geo.TaitBryan(H, P, R)
            rs = (rnd() for _ in range(100))
            for r in rs:
                Rd, Ri = tb.rot(0, 0, r * geo.TAU)
                self.assertTrue(np.allclose(Rd.dot(tb.rest), tb.rest))
                self.assertTrue(np.allclose(Ri.dot(tb.rest), tb.rest))
    def testH(self):
        """Heading properly refers to x, y or z."""
        v = np.array([1, 1, 1], dtype=complex)
        angle = geo.TAU / 4
        Rx, _ = geo.TaitBryan(0, 1, 2).rot(angle, 0, 0)
        Ry, _ = geo.TaitBryan(1, 2, 0).rot(angle, 0, 0)
        Rz, _ = geo.TaitBryan(2, 0, 1).rot(angle, 0, 0)
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
        Ry, _ = geo.TaitBryan(0, 1, 2).rot(0, angle, 0)
        Rz, _ = geo.TaitBryan(1, 2, 0).rot(0, angle, 0)
        Rx, _ = geo.TaitBryan(2, 0, 1).rot(0, angle, 0)
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
        Rz, _ = geo.TaitBryan(0, 1, 2).rot(0, 0, angle)
        Rx, _ = geo.TaitBryan(1, 2, 0).rot(0, 0, angle)
        Ry, _ = geo.TaitBryan(2, 0, 1).rot(0, 0, angle)
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
        R0, _ = geo.TaitBryan(0, 1, 2).rot(h, p, r)  # Right handed.
        R1, _ = geo.TaitBryan(0, 2, 1).rot(h, p, r)  # Left handed.
        R2, _ = geo.TaitBryan(1, 0, 2).rot(h, p, r)  # Counter clockwise.
        R3, _ = geo.TaitBryan(1, 2, 0).rot(h, p, r)  # Left handed.
        R4, _ = geo.TaitBryan(2, 0, 1).rot(h, p, r)  # Left handed.
        R5, _ = geo.TaitBryan(2, 1, 0).rot(h, p, r)  # Counter clockwise.
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
        # It's kinda funny. For a counter right handed tb, you can map the
        # components d, e and f directly onto the axes of the tb definition:
        # d=0, e=2, f=1.  However, for the left handed case it does not work.
        # Indeed, abc=012 for v1, abc=201 for v3 and abc=120 for v4.
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
                Rd, _ = tb.rot(h, p, r)
                vr = Rd.dot(tb.rest)
                hr, pr = tb.hp(vr)
                self.assertAlmostEqual(hr, h, 5)
                self.assertAlmostEqual(pr, p, 5)



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
