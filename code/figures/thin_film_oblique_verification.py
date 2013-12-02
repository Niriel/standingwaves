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
import sw.geometry as geo

Z0 = scipy.constants.value('characteristic impedance of vacuum')
TAU = np.pi * 2
n1 = 1.00
n2 = 1.50

Z1 = Z0 / n1
Z2 = Z0 / n2

d1 = 1.00
d2 = 10e-6

def setupA(f):
    tb = sw.geometry.TaitBryan(0, 1, 2)
    att = (TAU / 8, 0, 0)
    k1 = np.array([0, 0, 1])
    space1 = sw.networks.Distance(n1, d1, f)
    space2 = sw.networks.Distance(n1, d1, f)
    space3 = sw.networks.Distance(n1, d1, f)
    space4 = sw.networks.Distance(n1, d1, f)
    film, k2, k3, k4 = sw.networks.ThinFilmOblique(n1, n2, d2, f, tb, att, k1)
    networks = (film, space1, space2, space3, space4)
    #                  7
    #               [space2]
    #                  6
    #                  |
    #                  1
    # 5 [space1] 4 - 0 / 2 - 8 [space3] 9
    #                  3
    #                  |
    #                 10
    #               [space4]
    #                 11
    couplings = sw.ExpandCouplingsTo3d(({0, 4}, {1, 6}, {2, 8}, {3, 10}))
    nb_ports = 12 * 3  # *3 because 3D.
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
    a[5 * 3:5 * 3 + 3] = np.array([E, 0, 0])
    b = solve(a, c)
    rh = b[7 * 3:7 * 3 + 3]
    th = b[9 * 3:9 * 3 + 3]
    a[5 * 3:5 * 3 + 3] = np.array([0, E, 0])
    b = solve(a, c)
    rv = b[7 * 3:7 * 3 + 3]
    tv = b[9 * 3:9 * 3 + 3]
    return rh, rv, th, tv

def setupB(f):
    ao = TAU / 8  # Angle outside the film.
    tb = sw.geometry.TaitBryan(0, 1, 2)
    att1 = (ao, 0, 0)
    att2 = (TAU / 2 + ao, 0, 0)
    ai = geo.Snell(n1, n2, ao)
    li = d2 / np.cos(ai)  # Pathlength inside the film.
    lc = d2 / np.cos(ao)  # Pathlength to compensate for film thickness.
    lo = d1 - lc / 2  # Pathlength outside the film.
    na = tb.rot(*att1).dot(tb.rest)  # Normal to interface a.
    nb = tb.rot(*att2).dot(tb.rest)  # Normal to interface b.
    k1a = np.array([0, 0, 1])
    k1b = np.array([0, 0, -1])
    space_o = sw.networks.Distance(n1, lo, f)
    space_i = sw.networks.Distance(n2, li, f)
    inter_a, k2a, k3a, k4a = sw.networks.InterfaceOblique(n1, n2, na, k1a)
    inter_b, k2b, k3b, k4b = sw.networks.InterfaceOblique(n1, n2, nb, k1b)
#     inter_b, k2b, k3b, k4b = sw.networks.InterfaceOblique(n2, n1, na, k3a)
    networks = (inter_a, inter_b, space_i, space_i,
                space_o, space_o, space_o, space_o)
    couplings = sw.ExpandCouplingsTo3d(({0, 13}, {1, 15},
                                        {2, 8}, {3, 10},
                                        {4, 17}, {5, 19},
                                        {6, 9}, {7, 11}))
    nb_ports = 20 * 3  # *3 because 3D.
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
    #
    port_a = 12
    port_r = 14
    port_t = 16
    # Solve for H.
    a[port_a * 3:port_a * 3 + 3] = np.array([E, 0, 0])
    b = solve(a, c)
    rh = b[port_r * 3:port_r * 3 + 3]
    th = b[port_t * 3:port_t * 3 + 3]
    # Solve for V.
    a[port_a * 3:port_a * 3 + 3] = np.array([0, E, 0])
    b = solve(a, c)
    rv = b[port_r * 3:port_r * 3 + 3]
    tv = b[port_t * 3:port_t * 3 + 3]
    return rh, rv, th, tv

def plot():
    nb_points = 11
    f_range = 1000e9
    f_start = 0e9
    f_stop = f_start + f_range
    fs = np.linspace(f_start, f_stop, nb_points)
    rAhs = np.zeros(nb_points, dtype=complex)
    rAvs = np.zeros(nb_points, dtype=complex)
    tAhs = np.zeros(nb_points, dtype=complex)
    tAvs = np.zeros(nb_points, dtype=complex)
    rBhs = np.zeros(nb_points, dtype=complex)
    rBvs = np.zeros(nb_points, dtype=complex)
    tBhs = np.zeros(nb_points, dtype=complex)
    tBvs = np.zeros(nb_points, dtype=complex)
    rDhs = np.zeros(nb_points, dtype=complex)
    rDvs = np.zeros(nb_points, dtype=complex)
    tDhs = np.zeros(nb_points, dtype=complex)
    tDvs = np.zeros(nb_points, dtype=complex)

    for i, f in enumerate(fs):
        # P = EH   and   E=ZH   =>   P = E**2 / Z.
        rah, rav, tah, tav = setupA(f)
        rbh, rbv, tbh, tbv = setupB(f)
        rAhs[i] = np.linalg.norm(rah) ** 2 / Z1
        rAvs[i] = np.linalg.norm(rav) ** 2 / Z1
        tAhs[i] = np.linalg.norm(tah) ** 2 / Z1
        tAvs[i] = np.linalg.norm(tav) ** 2 / Z1
#         rAhs[i] = np.abs(rah[0]) ** 2 / Z1
#         rAvs[i] = np.abs(rav[2]) ** 2 / Z1
#         tAhs[i] = np.abs(tah[0]) ** 2 / Z1
#         tAvs[i] = np.abs(tav[1]) ** 2 / Z1

#         rBhs[i] = np.linalg.norm(rbh) ** 2 / Z1
#         rBvs[i] = np.linalg.norm(rbv) ** 2 / Z1
#         tBhs[i] = np.linalg.norm(tbh) ** 2 / Z1
#         tBvs[i] = np.linalg.norm(tbv) ** 2 / Z1
        rBhs[i] = np.abs(rbh[0]) ** 2 / Z1
        rBvs[i] = np.abs(rbv[2]) ** 2 / Z1
        tBhs[i] = np.abs(tbh[0]) ** 2 / Z1
        tBvs[i] = np.abs(tbv[1]) ** 2 / Z1

        rDhs[i] = np.abs(rbh[0] - rah[0])
        rDvs[i] = np.abs(rbv[2] - rav[2])
        tDhs[i] = np.abs(tbh[0] - tah[0])
        tDvs[i] = np.abs(tbv[0] - tav[0])

    xs = fs / 1.e9

    plt.subplot(3, 2, 1)
    plt.plot(xs, tAhs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film transmission"))
    plt.plot(xs, tBhs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces transmission"))
#     plt.plot(xs, tDhs)
    plt.title(style.latexT("Perpendicular"), fontsize=10)
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.ylim((.9, 1.1))
    plt.legend(loc='upper left', prop={"size":6})

    plt.subplot(3, 2, 2)
    plt.plot(xs, tAvs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film transmission"))
    plt.plot(xs, tBvs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces transmission"))
    plt.plot(xs, tDvs)
    plt.title(style.latexT("Parallel"), fontsize=10)
    plt.ylim((.99, 1.01))
    plt.legend(loc='upper right', prop={"size":6})

    plt.subplot(3, 2, 3)
    plt.plot(xs, rAhs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film reflection"))
    plt.plot(xs, rBhs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces reflection"))
#     plt.plot(xs, rDhs)
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.ylim((-.1, .1))
    plt.legend(loc='lower left', prop={"size":6})

    plt.subplot(3, 2, 4)
    plt.plot(xs, rAvs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film reflection"))
    plt.plot(xs, rBvs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces reflection"))
#     plt.plot(xs, rDvs)
    plt.ylim((-.01, .01))
    plt.legend(loc='lower right', prop={"size":6})

    plt.subplot(3, 2, 5)
    plt.plot(xs, rAhs + tAhs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film sum"))
    plt.plot(xs, rBhs + tBhs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces sum"))
    plt.ylim((.99, 1.01))
    plt.xlabel(style.latexT("Frequency [\\si{\\giga\\hertz}]"))
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.legend(loc='lower left', prop={"size":6})

    plt.subplot(3, 2, 6)
    plt.plot(xs, rAvs + tAvs, '-', color=style.COLORS_STD[1], label=style.latexT("Thin film sum"))
    plt.plot(xs, rBvs + tBvs, '--', color=style.COLORS_STD[2], label=style.latexT("Interfaces sum"))
    plt.ylim((.99, 1.01))
    plt.xlabel(style.latexT("Frequency [\\si{\\giga\\hertz}]"))
    plt.legend(loc='lower right', prop={"size":6})

    if USE_PDF:
        plt.savefig("thin_film_oblique_verification.pdf", bbox_inches='tight')
    else:
        plt.show()



def MakeFigure(width):
    with style.latex(width, scipy.constants.golden * 2 / 3):
        style.pretty()
        plot()

if __name__ == '__main__':
    MakeFigure(style.WIDTH_ARTICLE)
