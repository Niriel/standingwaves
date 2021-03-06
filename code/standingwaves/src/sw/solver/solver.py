'''
Created on Jun 18, 2013

@author: delforge
'''

import numpy as np
import scipy.linalg

def ExpandCouplingsTo3d(couplings):
    couplings3d = []
    for ca, cb in couplings:
        cax = ca * 3
        cay = cax + 1
        caz = cay + 1
        cbx = cb * 3
        cby = cbx + 1
        cbz = cby + 1
        couplings3d.extend([{cax, cbx}, {cay, cby}, {caz, cbz}])
    return couplings3d

def CheckUnusedCouplings(couplings, n, known_unused):
    """Return an exception if missing couplings, otherwise None."""
    unused = set(xrange(n))
    map(unused.discard, known_unused)
    for a, b in couplings:
        unused.discard(a)
        unused.discard(b)
    if len(unused) != 0:
        return ValueError("Some ports are not used: %r." % unused)
    return None

def CheckCouplingSanity(couplings, n):
    """Return exception if insane, otherwise None."""
    counters = {}
    for coupling in couplings:
        for port in coupling:
            if not (0 <= port < n):
                return ValueError("Port %i is outside the [0, %i] range." % (port, n - 1))
            counters[port] = counters.get(port, 0) + 1
            if counters[port] > 1:
                return ValueError("Port %i is used more than once." % port)
    return None

def FindInside(couplings):
    inside = set()
    map(inside.update, couplings)
    return inside

def FindOutside(inside, n):
    return set(range(n)).difference(inside)

def SeparateInsideOutside(inside, outside, n):
    insorted = sorted(list(inside))
    outsorted = sorted(list(outside))
    Q = np.zeros((n, n))
    for i, j in enumerate(outsorted):
        Q[i, j] = 1
    l = len(outsorted)
    for i, j in enumerate(insorted):
        Q[i + l, j] = 1
    return Q

def CoupleInputsToOutputs(inside, couplings):
    insorted = sorted(list(inside))
    m = len(inside)
    P = np.zeros((m, m))
    for portA, portB in couplings:
        indexA = insorted.index(portA)
        indexB = insorted.index(portB)
        P[indexA, indexB] = 1
        P[indexB, indexA] = 1
    return P

def GatherNetworks(networks):
    ns = [network.shape[0] for network in networks]
    n = sum(ns)
    S = np.zeros((n, n), dtype=complex)
    offset = 0
    for ni, network in zip(ns, networks):
        S[offset:offset + ni, offset:offset + ni] = network
        offset += ni
    return S

def FactorizeSiiP(SiiP):
    m = SiiP.shape[0]
    if m == 0:
        return None
    I = np.identity(m, dtype=complex)
    M = I - SiiP
    return scipy.linalg.lu_factor(M, False)

def SeparateMatrixRegions(S1, no):
    S1oo = S1[:no, :no]
    S1oi = S1[:no, no:]
    S1io = S1[no:, :no]
    S1ii = S1[no:, no:]
    return S1oo, S1oi, S1io, S1ii

def SeparateVectorRegions(v, no):
    return v[:no], v[no:]

def SolveCouplings(couplings, n):
    inside = FindInside(couplings)
    outside = FindOutside(inside, n)
    Q = SeparateInsideOutside(inside, outside, n)
    P = CoupleInputsToOutputs(inside, couplings)
    no = len(outside)
    return P, Q, no

def SolveNetworks((P, Q, no), networks):
    S = GatherNetworks(networks)
    if Q.shape != S.shape:
        raise ValueError("Declared number of ports (%i) does not match the sum of the ports of each network (%i)." % (Q.shape[0], P.shape[0]))
    S1 = Q.dot(S).dot(Q.T)
    S1oo, S1oi, S1io, S1ii = SeparateMatrixRegions(S1, no)
    S1iiP = S1ii.dot(P)
    LU = FactorizeSiiP(S1iiP)
    return S1oo, S1oi, S1io, LU

def SolveInputs(P, S1oo, S1oi, S1io, LU, a1o, c1o, c1i):
    if LU is None:
        # There is no inside port.
        b1 = S1oo.dot(a1o) + c1o
    else:
        entering = S1io.dot(a1o) + c1i
        b1i = scipy.linalg.lu_solve(LU, entering)
        b1o = S1oo.dot(a1o) + S1oi.dot(P).dot(b1i) + c1o
        b1 = np.concatenate((b1o, b1i))
    return b1

def Solveb((P, Q, no), (S1oo, S1oi, S1io, LU), a, c):
    a1 = Q.dot(a)
    c1 = Q.dot(c)
    a1o, _ = SeparateVectorRegions(a1, no)
    c1o, c1i = SeparateVectorRegions(c1, no)
    b1 = SolveInputs(P, S1oo, S1oi, S1io, LU, a1o, c1o, c1i)
    b = (Q.T).dot(b1)
    return b

def Solver(n, couplings):
    err = CheckCouplingSanity(couplings, n)
    if err is not None:
        raise err
    solvedcouplings = SolveCouplings(couplings, n)
    def sendNetworks(networks):
        solvednetworks = SolveNetworks(solvedcouplings, networks)
        def solve(a, c):
            return Solveb(solvedcouplings, solvednetworks, a, c)
        return solve
    return sendNetworks


