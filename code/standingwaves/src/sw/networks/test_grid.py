'''
Created on Jul 23, 2013

@author: delforge
'''
import unittest
import numpy as np
import sw.geometry as geo
import grid

class TestRT(unittest.TestCase):
    def testPerfectNormal(self):
        a = 1e-9  # Very tiny, perfect.
        d = 10e-9  # Because a<<d required.
        s = 1e10  # Extremely good conductor, perfect.
        f = 1e12  # 1 THz, typical HIFI frequency.
        propdir = np.array([0, 0, 1])  # Normal incidence.
        R, T = grid._GridRT(a, d, s, f, propdir)
        # Field parallel to the wires.
        # ----------------------------
        ei = np.array([1, 0, 0])
        er = R.dot(ei)
        et = T.dot(ei)
        # Mostly reflected, almost nothing transmitted.
        # The reflected part should have its sign flipped because of the reflection
        # on a higher index.
        self.assertTrue(np.allclose(er, -ei, 0, 1.e-3))
        self.assertTrue(np.allclose(et, np.array([0, 0, 0]), 0, 1.e-3))
        # Ohmic losses in the wires are very low, energy mostly conserved.
        Pi = np.linalg.norm(ei) ** 2
        Pr = np.linalg.norm(er) ** 2
        Pt = np.linalg.norm(et) ** 2
        self.assertTrue(0 <= Pi - Pr - Pt <= 1.e-3)
        # Field perpendicular to the wires.
        # ---------------------------------
        ei = np.array([0, 1, 0])
        er = R.dot(ei)
        et = T.dot(ei)
        # Mostly transmitted, almost nothing reflected.
        self.assertTrue(np.allclose(er, np.array([0, 0, 0]), 0, 1.e-3))
        self.assertTrue(np.allclose(et, ei, 0, 1.e-3))
        # Ohmic losses in the wires are very low.
        Pi = np.linalg.norm(ei) ** 2
        Pr = np.linalg.norm(er) ** 2
        Pt = np.linalg.norm(et) ** 2
        self.assertTrue(0 <= Pi - Pr - Pt <= 1.e-3)
    def compare(self, G, a, expected):
            b = G.dot(a)
#             print np.abs(b)
#             print np.abs(expected)
            self.assertTrue(np.allclose(b, expected, 0, 1.e-3))
    def testAttitude1(self):
        a = 1e-9  # Very tiny, perfect.
        d = 10e-9  # Because a<<d required.
        s = 1e10  # Extremely good conductor, perfect.
        f = 1e12  # 1 THz, typical HIFI frequency.
        tb = geo.TaitBryan(0, 1, 2)
        # Rotate the grid around its normal so that the wires are y-oriented.
        attitude = (0, 0, geo.TAU / 4)
        k1 = np.array([0, 0, 1])
        G = grid.Grid(a, d, s, f, tb, attitude, k1)
        # Input x field on port 1.
        # No reflection on 2, transmission on 3.
        a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input x field on port 2.
        # No reflection on 1, transmission on 4.
        a = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        self.compare(G, a, b)
        # Input x field on port 3.
        # No reflection on 4, transmission on 1.
        a = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input x field on port 4.
        # No reflection on 3, transmission on 2.
        a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        b = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input y field on port 1.
        # The reflected wave flips sign.
        # Reflection on 2, no transmission on 3.
        a = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input y field on port 2.
        # Reflection on 1, no transmission on 4.
        a = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input y field on port 3.
        # Reflection on 4, no transmission on 1.
        a = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0])
        self.compare(G, a, b)
        # Input y field on port 4.
        # Reflection on 3, no transmission on 2.
        a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0])
        self.compare(G, a, b)
    def testAttitude2(self):
        a = 1e-9  # Very tiny, perfect.
        d = 10e-9  # Because a<<d required.
        s = 1e10  # Extremely good conductor, perfect.
        f = 1e12  # 1 THz, typical HIFI frequency.
        tb = geo.TaitBryan(0, 1, 2)
        # Two rotations.  The grid is now in the plane yz and the wires along z.
        attitude = (geo.TAU / 4, 0, geo.TAU / 4)
        k1 = np.array([0, -1, 0])
        G = grid.Grid(a, d, s, f, tb, attitude, k1)
        # Input x field on port 1.
        # No reflection on 2, transmission on 3.
        a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input x field on port 2.
        # No reflection on 1, transmission on 4.
        a = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        self.compare(G, a, b)
        # Input x field on port 3.
        # No reflection on 4, transmission on 1.
        a = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        b = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input x field on port 4.
        # No reflection on 3, transmission on 2.
        a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
        b = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input z field on port 1.
        # Reflection on 2, no transmission on 3.
        a = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input z field on port 2.
        # Reflection on 1, no transmission on 4.
        a = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # Input z field on port 3.
        # Reflection on 4, no transmission on 1.
        a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1])
        self.compare(G, a, b)
        # Input z field on port 4.
        # Reflection on 3, no transmission on 2.
        a = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
        b = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0])
        self.compare(G, a, b)
    def testAttitude3(self):
        a = 1e-9  # Very tiny, perfect.
        d = 10e-9  # Because a<<d required.
        s = 1e10  # Extremely good conductor, perfect.
        f = 1e12  # 1 THz, typical HIFI frequency.
        tb = geo.TaitBryan(0, 1, 2)
        attitude = (geo.TAU / 8, 0, 0)
        k1 = np.array([0, 0, 1])
        G = grid.Grid(a, d, s, f, tb, attitude, k1)
        # I send x on 1.
        # Reflection on 2, no transmission on 3.
        a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0])
        self.compare(G, a, b)
        # I send y on 1.
        # No reflection on 2, transmission on 3.
        a = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        self.compare(G, a, b)
        # I send both x and y on 1.
        # x should be reflected on 2 and y transmitted on 3.
        f = np.sqrt(2) * .5
        a = np.array([f, f, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        b = np.array([0, 0, 0, -f, 0, 0, 0, f, 0, 0, 0, 0])
        self.compare(G, a, b)
    def testAttitude4(self):
        # Diplexer beam splitter configuration. The angle of incidence is 45
        # degrees. In addition, the wires APPEAR at 45 degrees relatively to the
        # polarization of the wave (angle=atan(sqrt(2)).
        a = 1e-9  # Very tiny, perfect.
        d = 10e-9  # Because a<<d required.
        s = 1e10  # Extremely good conductor, perfect.
        f = 1e12  # 1 THz, typical HIFI frequency.
        tb = geo.TaitBryan(0, 1, 2)
        attitude = (geo.TAU / 8, 0, np.arctan(np.sqrt(2)))
        k1 = np.array([0, 0, 1])
        G = grid.Grid(a, d, s, f, tb, attitude, k1)
        # I send x.
        # I expect 50% of reflection on 2 and 50% of transmission on 3.
        # For each of these ports, I expect 50% of energy in each polarization.
        # Plus xy becomes x(-z) when reflected,
        # So, 25% in 2x, 25 in 2z, 25 in 3x, 25 in 3y.
        a = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        e = np.array([0, 0, 0, .25, 0, .25, .25, .25, 0, 0, 0, 0])
        b = np.abs((G.dot(a))) ** 2  # Work in power.
        print e
        print b
        print np.abs(b - e)
        self.assertTrue(np.allclose(b, e, 0, 1.e-2))


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
