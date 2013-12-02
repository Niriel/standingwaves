import unittest
import numpy as np
import interface
from sw import geometry as geo

TAU = 2 * np.pi

class TestInterfaceGeometry(unittest.TestCase):
    def testAngleOfIncidence(self):
        tb = geo.TaitBryan(2, 0, 1)  # Like in Panda3d.
        n = tb.rotH(np.radians(60)).dot(tb.rest)
        k1 = tb.rotH(np.radians(30)).dot(tb.rest)
        ig = interface._InterfaceGeometry(1, 1.5, n, k1)
        self.assertAlmostEqual(ig.theta_a, np.radians(30))
    def testProjections(self):
        tb = geo.TaitBryan(2, 0, 1)  # Like in Panda3d.
        n = tb.rotH(np.radians(60)).dot(tb.rest)
        k1 = tb.rotH(np.radians(30)).dot(tb.rest)
        ig = interface._InterfaceGeometry(1, 1.5, n, k1)
        # Verify field projection.
        ep = np.array([1, 1, 0])
        es = np.array([0, 0, 1])
        self.assertTrue(np.allclose(ig.P.dot(ep), ep))
        self.assertTrue(np.allclose(ig.S.dot(es), es))
    def testPort12(self):
        tb = geo.TaitBryan(2, 0, 1)  # Like in Panda3d.
        n = tb.rotH(np.radians(60)).dot(tb.rest)
        # We want a reflected beam in the direction +x.
        # Our incoming beam is almost +y.
        k1 = tb.rotH(np.radians(30)).dot(tb.rest)
        # Here the indices do not matter, we just test the reflection.
        ig = interface._InterfaceGeometry(1, 1.5, n, k1)
        # Verify direction of propagation.
        hope = np.array([1, 0, 0])
        self.assertTrue(np.allclose(ig.k2, hope))
        # Verify field rotations.
        ep1 = tb.rotH(np.radians(90)).dot(k1)
        es = np.array([0, 0, 1])
        ep2 = ig.R21.dot(ep1)
        hope = np.array([0, -1, 0])
        self.assertTrue(np.allclose(ep2, hope))
        self.assertTrue(np.allclose(ig.R12.dot(ep2), ep1))
        self.assertTrue(np.allclose(ig.R21.dot(es), es))
        self.assertTrue(np.allclose(ig.R12.dot(es), es))
    def testPort31(self):
        tb = geo.TaitBryan(2, 0, 1)  # Like in Panda3d.
        n = tb.rotH(np.radians(30)).dot(tb.rest)
        # n points a bit to the left. -x, +y.
        k1 = np.array([0, 1, 0])
        n1 = np.sqrt(3)
        n2 = 1
        ig = interface._InterfaceGeometry(n1, n2, n, k1)
        # Verify direction of propagation.
        # k3 = 30 - 60 = -30 degrees.
        # k4 = 30 + 60 = 90 degrees.
        self.assertTrue(np.allclose(ig.k3, tb.rotH(np.radians(-30)).dot(tb.rest)))
        self.assertTrue(np.allclose(ig.k4, tb.rotH(np.radians(90)).dot(tb.rest)))
        # Verify field rotations.
        ep1 = tb.rotH(np.radians(90)).dot(k1)
        es = np.array([0, 0, 1])
        ep3_hope = tb.rotH(np.radians(90 - 30)).dot(tb.rest)
        ep3 = ig.R31.dot(ep1)
        self.assertTrue(np.allclose(ep3, ep3_hope))
        self.assertTrue(np.allclose(ig.R31.dot(es), es))
        ep1_hope = ep1
        ep1 = ig.R13.dot(ep3)
        self.assertTrue(np.allclose(ep1, ep1_hope))
        self.assertTrue(np.allclose(ig.R13.dot(es), es))
    def testPort43(self):
        tb = geo.TaitBryan(2, 0, 1)  # Like in Panda3d.
        n = tb.rotH(np.radians(60)).dot(tb.rest)
        # We want a reflected beam in the direction +x.
        # Our incoming beam is almost +y.
        k1 = tb.rotH(np.radians(30)).dot(tb.rest)
        n1 = np.sqrt(3)
        n2 = 1
        ig = interface._InterfaceGeometry(n1, n2, n, k1)
        # Verify directions of propagation.
        k2_hope = -tb.rotH(np.radians(2 * 30)).dot(k1)
        k3_hope = np.array([0, 1, 0])
        k4_hope = tb.rotH(np.radians(120)).dot(tb.rest)
        self.assertTrue(np.allclose(ig.k2, k2_hope))
        self.assertTrue(np.allclose(ig.k3, k3_hope))
        self.assertTrue(np.allclose(ig.k4, k4_hope))
        # Verify field rotations.
        ep4 = tb.rotH(np.radians(90)).dot(ig.k4)
        ep3_hope = np.array([-1, 0, 0])
        ep2_hope = tb.rotH(np.radians(90)).dot(-ig.k2)
        ep3 = ig.R34.dot(ep4)
        ep2 = ig.R24.dot(ep4)
        self.assertTrue(np.allclose(ep3, ep3_hope))
        self.assertTrue(np.allclose(ep2, ep2_hope))
        ep4_hope = ep4
        ep4 = ig.R43.dot(ep3)
        self.assertTrue(np.allclose(ep4, ep4_hope))
        ep4 = ig.R42.dot(ep2)
        self.assertTrue(np.allclose(ep4, ep4_hope))



if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
