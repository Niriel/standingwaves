'''
Created on Jul 23, 2013

@author: delforge
'''
import unittest
import numpy as np
import grid

class TestRT(unittest.TestCase):
    def testPerfectNormal(self):
        a = 1e-9  # Very tiny, perfect.
        d = 10e-9  # Because a<<d required.
        s = 1e10  # Extremely good conductor, perfect.
        f = 1e12  # 1 THz, typical HIFI frequency.
        propdir = np.array([0, 0, 1])  # Normal incidence.
        R, T = grid.RT(a, d, s, f, propdir)
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


if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
