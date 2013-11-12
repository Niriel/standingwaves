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

def CheckCouplingSanity(couplings, n):
    """No return value, raise exception if insane."""
    counters = {}
    for coupling in couplings:
        for port in coupling:
            if not (0 <= port < n):
                raise ValueError("Port %i is outside the [0, %i] range." % (port, n - 1))
            counters[port] = counters.get(port, 0) + 1
            if counters[port] > 1:
                raise ValueError("Port %i is used more than once." % port)

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

def SolveNetworks(P, Q, no, networks):
    S = GatherNetworks(networks)
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

def Solver(n, couplings):
    CheckCouplingSanity(couplings, n)
    P, Q, no = SolveCouplings(couplings, n)
    def sendNetworks(networks):
        S1oo, S1oi, S1io, LU = SolveNetworks(P, Q, no, networks)
        def solve(a, c):
            a1 = Q.dot(a)
            c1 = Q.dot(c)
            a1o, _ = SeparateVectorRegions(a1, no)
            c1o, c1i = SeparateVectorRegions(c1, no)
            b1 = SolveInputs(P, S1oo, S1oi, S1io, LU, a1o, c1o, c1i)
            b = (Q.T).dot(b1)
            return b
        return solve
    return sendNetworks


