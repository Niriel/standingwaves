from __future__ import division
USE_PDF = True
import numpy as np
import matplotlib
if USE_PDF:
    matplotlib.use('pdf')
import matplotlib.pyplot as plt
import scipy.constants
import style.my_style as style
import sw

Z0 = scipy.constants.value('characteristic impedance of vacuum')

n1 = 1.00
n2 = 1.50
n3 = 1.25

Z1 = Z0 / n1
Z2 = Z0 / n2
Z3 = Z0 / n3

d1 = 1.00
d2 = 10e-6
d3 = 1.00

print (scipy.constants.speed_of_light / n2) / (2 * d2) / 1.e9

def setupA(f):
    tb = sw.geometry.TaitBryan(0, 1, 2)
    space1 = sw.networks.Distance(n1, d1, f)
    film = sw.networks.ThinFilmNormal(n1, n2, n3, d2, f, tb, (0, 0, 0))
    space3 = sw.networks.Distance(n3, d3, f)
    networks = (space1, film, space3)
    # 0 [space1] 1 - 2 [film] 3 - 4 [space2] 5.
    couplings = sw.ExpandCouplingsTo3d(({1, 2}, {3, 4}))
    nb_ports = 6 * 3  # *3 because 3D.
    presolve = sw.Solver(nb_ports, couplings)
    solve = presolve(networks)
    # Inputs.
    a = np.zeros(nb_ports, dtype=complex)
    c = np.zeros_like(a)
    # E = ZH    Ohm's law.
    # P = EH = E**2/Z
    # E**2 = ZP
    # E = sqrt ZP
    P = 1
    E = (Z1 * P) ** .5
    a[0:3] = np.array([E, 0, 0])
    b = solve(a, c)
    b0 = b[0 * 3:0 * 3 + 3]
    b5 = b[5 * 3:5 * 3 + 3]
    return b0[0], b5[0]

def setupB(f):
    space1 = sw.networks.Distance(n1, d1 - d2 / 2, f)
    space2 = sw.networks.Distance(n2, d2, f)
    space3 = sw.networks.Distance(n3, d3 - d2 / 2, f)
    inter1 = sw.networks.InterfaceNormal(n1, n2)
    inter3 = sw.networks.InterfaceNormal(n2, n3)
    networks = (space1, inter1, space2, inter3, space3)
    # 0 [space1] 1 - 2 [inter1] 3 - 4 [space2] 5 - 6 [inter3] 7 - 8 [space3] 9.
    couplings = sw.ExpandCouplingsTo3d(({1, 2}, {3, 4}, {5, 6}, {7, 8}))
    nb_ports = 10 * 3  # *3 because 3D.
    presolve = sw.Solver(nb_ports, couplings)
    solve = presolve(networks)
    # Inputs.
    a = np.zeros(nb_ports, dtype=complex)
    c = np.zeros_like(a)
    # E = ZH    Ohm's law.
    # P = EH = E**2/Z
    # E**2 = ZP
    # E = sqrt ZP
    P = 1
    E = (Z1 * P) ** .5
    a[0:3] = np.array([E, 0, 0])
    b = solve(a, c)
    b0 = b[0 * 3:0 * 3 + 3]
    b9 = b[9 * 3:9 * 3 + 3]
    return b0[0], b9[0]

def plot():
    nb_points = 101
    f_range = 1000e9
    f_start = 0e9
    f_stop = f_start + f_range
    fs = np.linspace(f_start, f_stop, nb_points)
    rAs = np.zeros(nb_points, dtype=complex)
    tAs = np.zeros(nb_points, dtype=complex)
    rBs = np.zeros(nb_points, dtype=complex)
    tBs = np.zeros(nb_points, dtype=complex)
    for i, f in enumerate(fs):
        rAs[i], tAs[i] = setupA(f)
        rBs[i], tBs[i] = setupB(f)
    # P = EH   and   E=ZH   =>   P = E**2 / Z.
    yrAs = np.abs(rAs) ** 2 / Z3
    ytAs = np.abs(tAs) ** 2 / Z3
    yrBs = np.abs(rBs) ** 2 / Z3
    ytBs = np.abs(tBs) ** 2 / Z3
    xs = fs / 1.e9
    plt.subplot(2, 1, 1)
    plt.plot(xs, ytAs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film model"))
    plt.plot(xs, ytBs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces model"))
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.annotate(style.latexT("Transmitted"), xy=(150, np.mean(ytAs)), verticalalignment='center', size='x-small')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(xs, yrAs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film model"))
    plt.plot(xs, yrBs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces model"))
    plt.xlabel(style.latexT("Frequency [\\si{\\giga\\hertz}]"))
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.annotate(style.latexT("Reflected"), xy=(150, np.mean(yrAs)), verticalalignment='center', size='x-small')
    plt.legend(loc='lower right')
    if USE_PDF:
        plt.savefig("thin_film_normal_verification.pdf", bbox_inches='tight')
    else:
        plt.show()



def MakeFigure(width):
    with style.latex(width):
        style.pretty()
        plot()

if __name__ == '__main__':
    MakeFigure(style.WIDTH_ARTICLE)
