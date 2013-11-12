import unittest
import numpy as np
import scipy.linalg
import solver

class TestExpandCouplingsTo3d(unittest.TestCase):
    def testTrippleSize(self):
        couplings1d = [{10, 20}, {40, 30}]
        couplings3d = solver.ExpandCouplingsTo3d(couplings1d)
        expected = [{30, 60}, {31, 61}, {32, 62}, {90, 120}, {91, 121}, {92, 122}]
        self.assertTrue(all((c == e for c, e in zip(couplings3d, expected))))

class TestFindInside(unittest.TestCase):
    def testSane(self):
        couplings = [{1, 2}, {3, 4}]
        inside = solver.FindInside(couplings)
        self.assertEquals(inside, {1, 2, 3, 4})
    def testNoInside(self):
        inside = solver.FindInside([])
        self.assertEquals(inside, set([]))

class TestFindOutside(unittest.TestCase):
    def testFindOutside(self):
        inside = {1, 2, 3, 4}
        n = 6
        outside = solver.FindOutside(inside, n)
        self.assertEquals(outside, {0, 5})
    def testNoInside(self):
        inside = set([])
        n = 4
        outside = solver.FindOutside(inside, n)
        self.assertEquals(outside, {0, 1, 2, 3})
    def testNoOutside(self):
        inside = {0, 1, 2, 3}
        n = 4
        outside = solver.FindOutside(inside, n)
        self.assertEquals(outside, set([]))
    def testNothing(self):
        inside = set([])
        n = 0
        outside = solver.FindOutside(inside, n)
        self.assertEquals(outside, set([]))

class TestSeparateInsideOutside(unittest.TestCase):
    def testSane(self):
        inside = {1, 2, 3, 4}
        outside = {0, 5}
        n = 6
        Q = solver.SeparateInsideOutside(inside, outside, n)
        E = np.array([[1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1],
                      [0, 1, 0, 0, 0, 0],
                      [0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1, 0]])
        self.assertTrue(np.array_equal(Q, E))
    def testNoInside(self):
        inside = set([])
        outside = {0, 1, 2}
        n = 3
        Q = solver.SeparateInsideOutside(inside, outside, n)
        E = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        self.assertTrue(all((q == e for q, e in zip(Q.flat, E.flat))))
    def testNoOutside(self):
        inside = {0, 1, 2}
        outside = set([])
        n = 3
        Q = solver.SeparateInsideOutside(inside, outside, n)
        E = np.array([[1, 0, 0],
                      [0, 1, 0],
                      [0, 0, 1]])
        self.assertTrue(np.array_equal(Q, E))
    def testNothing(self):
        inside = set([])
        outside = set([])
        n = 0
        Q = solver.SeparateInsideOutside(inside, outside, n)
        self.assertEquals(Q.shape, (0, 0))

class TestCoupleInputsToOutputs(unittest.TestCase):
    def testSane(self):
        couplings = [{1, 2}, {3, 4}]
        inside = {1, 2, 3, 4}
        P = solver.CoupleInputsToOutputs(inside, couplings)
        E = np.array([[0, 1, 0, 0],
                      [1, 0, 0, 0],
                      [0, 0, 0, 1],
                      [0, 0, 1, 0]])
        self.assertTrue(np.array_equal(P, E))
    def testNoInside(self):
        couplings = []
        inside = set([])
        P = solver.CoupleInputsToOutputs(inside, couplings)
        self.assertEquals(P.shape, (0, 0))

class TestGatherNetworks(unittest.TestCase):
    def testSane(self):
        A = np.array([[1, 2],
                      [3, 4]])
        B = np.array([[5, 6, 7],
                      [8, 9, 10],
                      [11, 12, 13]])
        S = solver.GatherNetworks([A, B])
        E = np.array([[1, 2, 0, 0, 0],
                      [3, 4, 0, 0, 0],
                      [0, 0, 5, 6, 7],
                      [0, 0, 8, 9, 10],
                      [0, 0, 11, 12, 13]])
        self.assertTrue(np.array_equal(S, E))
    def testOne(self):
        A = np.array([[1, 2],
                      [3, 4]])
        S = solver.GatherNetworks([A])
        E = np.array([[1, 2],
                      [3, 4]])
        self.assertTrue(np.array_equal(S, E))
    def testZero(self):
        S = solver.GatherNetworks([])
        self.assertEquals(S.shape, (0, 0))

class TestFactorizeSiiP(unittest.TestCase):
    def testSane(self):
        ISiiP = np.array([[1, 2],
                          [3, 4]])
        SiiP = np.identity(2, dtype=complex) - ISiiP
        LU = solver.FactorizeSiiP(SiiP)
        # Even though the matrix is integer, the LU decomposition won't be.
        # Maybe I should spend time trying to make an integer LU.  Note that in
        # any case it will be stored as a matrix of complex numbers.
        e = np.array([[1], [2]])  # What we *e*xpect to find.
        y = np.array([[5], [11]])  # ISiiP.dot(e)
        x = scipy.linalg.lu_solve(LU, y)  # Solve ISiiP x = y.
        self.assertTrue(np.array_equal(x, e))
    def testNoInside(self):
        ISiiP = np.zeros(shape=(0, 0))
        SiiP = np.identity(0, dtype=complex) - ISiiP
        LU = solver.FactorizeSiiP(SiiP)
        self.assertEquals(LU, None)

class TestSeparateVectorRegions(unittest.TestCase):
    def testSane(self):
        vo, vi = solver.SeparateVectorRegions(np.array([1, 2, 3]), 1)
        self.assertTrue(np.array_equal(vo, np.array([1])))
        self.assertTrue(np.array_equal(vi, np.array([2, 3])))
    def testNoOutside(self):
        vo, vi = solver.SeparateVectorRegions(np.array([1, 2, 3]), 0)
        self.assertTrue(np.array_equal(vo, np.array([])))
        self.assertTrue(np.array_equal(vi, np.array([1, 2, 3])))
    def testNoInside(self):
        vo, vi = solver.SeparateVectorRegions(np.array([1, 2, 3]), 3)
        self.assertTrue(np.array_equal(vo, np.array([1, 2, 3])))
        self.assertTrue(np.array_equal(vi, np.array([])))
    def testNothing(self):
        vo, vi = solver.SeparateVectorRegions(np.array([]), 0)
        self.assertTrue(np.array_equal(vo, np.array([])))
        self.assertTrue(np.array_equal(vi, np.array([])))

class TestSeparateMatrixRegions(unittest.TestCase):
    def testSane(self):
        S1 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
        no = 1
        S1oo, S1oi, S1io, S1ii = solver.SeparateMatrixRegions(S1, no)
        E = np.array([[1]])
        self.assertTrue(np.array_equal(S1oo, E))
        E = np.array([[2, 3]])
        self.assertTrue(np.array_equal(S1oi, E))
        E = np.array([[4],
                      [7]])
        self.assertTrue(np.array_equal(S1io, E))
        E = np.array([[5, 6],
                      [8, 9]])
        self.assertTrue(np.array_equal(S1ii, E))
    def testNoInside(self):
        S1 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
        no = 3
        S1oo, S1oi, S1io, S1ii = solver.SeparateMatrixRegions(S1, no)
        E = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        self.assertTrue(np.array_equal(S1oo, E))
        E = np.zeros(shape=(3, 0))
        self.assertTrue(np.array_equal(S1oi, E))
        E = np.zeros(shape=(0, 3))
        self.assertTrue(np.array_equal(S1io, E))
        E = np.zeros(shape=(0, 0))
        self.assertTrue(np.array_equal(S1ii, E))
    def testNoOutside(self):
        S1 = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9]])
        no = 0
        S1oo, S1oi, S1io, S1ii = solver.SeparateMatrixRegions(S1, no)
        E = np.zeros(shape=(0, 0))
        self.assertTrue(np.array_equal(S1oo, E))
        E = np.zeros(shape=(0, 3))
        self.assertTrue(np.array_equal(S1oi, E))
        E = np.zeros(shape=(3, 0))
        self.assertTrue(np.array_equal(S1io, E))
        E = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]])
        self.assertTrue(np.array_equal(S1ii, E))
    def testNothing(self):
        S1 = np.zeros(shape=(0, 0))
        no = 0
        S1oo, S1oi, S1io, S1ii = solver.SeparateMatrixRegions(S1, no)
        E = np.zeros(shape=(0, 0))
        self.assertTrue(np.array_equal(S1oo, E))
        self.assertTrue(np.array_equal(S1oi, E))
        self.assertTrue(np.array_equal(S1io, E))
        self.assertTrue(np.array_equal(S1ii, E))

class TestSolve(unittest.TestCase):
    def testSane(self):
        network1 = np.array([[.1, .5],
                             [.5, 0]])
        network2 = np.array([[0, .4],
                             [.4, 0]])
        networks = [network1, network2]
        couplings = [{1, 2}]
        n = sum((n.shape[0] for n in networks))
        solvedCouplings = solver.Solver(n, couplings)
        solvedNetworks = solvedCouplings(networks)
        a = np.array([1, 0, 0, 0])
        c = np.array([0, 3, 0, 0])
        b = solvedNetworks(a, c)
        e = np.array([.1, 3.5, 0, 1.4])
        self.assertTrue(np.allclose(b, e))
    def testNoInside(self):
        network1 = np.array([[.1, .5],
                             [.5, 0]])
        networks = [network1]
        couplings = []
        n = sum([n.shape[0] for n in networks])
        solvedCouplings = solver.Solver(n, couplings)
        solvedNetworks = solvedCouplings(networks)
        a = np.array([1, 0])
        c = np.array([0, 3])
        b = solvedNetworks(a, c)
        e = np.array([.1, 3.5])
        self.assertTrue(np.allclose(b, e))
    def testNoOutside(self):
        network1 = np.array([[.1]])
        network2 = np.array([[.1]])
        networks = [network1, network2]
        couplings = [{0, 1}]
        n = sum([n.shape[0] for n in networks])
        solvedCouplings = solver.Solver(n, couplings)
        solvedNetworks = solvedCouplings(networks)
        a = np.array([0, 0])
        c = np.array([1, 0])
        b = solvedNetworks(a, c)
        # Port 0: Start with 1, and each back and forth divides by 100.
        # Port 1: same, starting with 0.1.
        e = np.array([1.01010101010101010101, 0.101010101010101010101])
        self.assertTrue(np.allclose(b, e))
    def testNothing(self):
        networks = []
        couplings = []
        n = sum([n.shape[0] for n in networks])
        solvedCouplings = solver.Solver(n, couplings)
        solvedNetworks = solvedCouplings(networks)
        a = np.array([])
        c = np.array([])
        b = solvedNetworks(a, c)
        self.assertTrue(np.array_equal(b, np.zeros(shape=(0,))))

if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
