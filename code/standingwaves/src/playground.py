import sw
import numpy as np


def compute():
    frequencies = np.linspace(1000e9, 1004e9, 1000)
    tb = sw.geometry.TaitBryan(0, 1, 2)
    n1 = 1
    n2 = 2
    thickness = 10e-6
    attitude = [sw.geometry.TAU / 8, 0, 0]
    k1 = np.array([0, 0, 1])

    black = sw.networks.Mirror(0)
    couplings = [{1, 4}, {3, 5}]
    couplings = sw.ExpandCouplingsTo3d(couplings)
    presolve = sw.Solver(6 * 3, couplings)

    a1o = np.array([1, 0, 0, 0, 0, 0])
    def runModel(frequency):
        film, k2, k3, k4 = sw.networks.ThinFilm(n1, n2, thickness, frequency, tb, attitude, k1)
        networks = [film, black, black]
        solver = presolve(networks)
        b = solver(a1o)
        return np.abs(b[2 * 3]) ** 2
    y = np.array(map(runModel, frequencies), dtype=float)
    return frequencies, y


x = y = None  # To shut the IDE up.
try:
    import cProfile as profile
except ImportError:
    import profile
profile.run('x, y = compute()', 'profile.prf')
import pstats
pstats.Stats('profile.prf').sort_stats('time').print_stats(50)

import matplotlib.pyplot as plt
plt.plot(x / 1.e9, y)
plt.show()
