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

def CoupleInputsToOutputs(couplings):
    inside = FindInside(couplings) # TODO: remove, duplicated work. 
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
    I = np.identity(m, dtype=complex)
    M = I - SiiP
    return scipy.linalg.lu_factor(M, False)

def Solver(n, couplings):
    inside = FindInside(couplings)
    outside = FindOutside(inside, n)
    Q = SeparateInsideOutside(inside, outside, n)
    P = CoupleInputsToOutputs(couplings)
    Qt = Q.T
    def sendNetworks(networks):
        S = GatherNetworks(networks)
        S1 = Q.dot(S).dot(Qt)
        lim = len(outside)
        S1oo = S1[:lim, :lim]
        S1oi = S1[:lim, lim:]
        S1io = S1[lim:, :lim]
        S1ii = S1[lim:, lim:]
        S1iiP = S1ii.dot(P)
        LU = FactorizeSiiP(S1iiP)
        def solve(a1o):
            entering = S1io.dot(a1o)
            b1i = scipy.linalg.lu_solve(LU, entering)
            b1o = S1oo.dot(a1o) + S1oi.dot(P).dot(b1i)
            b1 = np.concatenate((b1o, b1i))
            b = (Q.T).dot(b1)
            return b
        return solve
    return sendNetworks


