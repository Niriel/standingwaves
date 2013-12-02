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
TAU = np.pi * 2

def setup(n1, n2, a):
    tb = sw.geometry.TaitBryan(0, 1, 2)
    att1 = (a, 0, 0)
    na = tb.rot(*att1).dot(tb.rest)  # Normal to interface a.
    k1 = np.array([0, 0, 1])
    inter, k2, k3, k4 = sw.networks.InterfaceOblique(n1, n2, na, k1)

    networks = (inter,)
    couplings = sw.ExpandCouplingsTo3d([])
    nb_ports = 4 * 3  # *3 because 3D.
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
    Z1 = Z0 / n1  # True for dielectrics only.
    E = (Z1 * P) ** .5
    #
    port_a = 0
    port_r = 1
    port_t = 2
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
    fs_normal = np.linspace(1, 90, 90)
    fs_critical = np.linspace(41.8103, 41.8104, 10)  # High resolution around critical angle.
    fs = np.concatenate((fs_normal, fs_critical))
    fs.sort()
    fs = np.radians(fs)
    nb_points = len(fs)
    rAhs = np.zeros(nb_points, dtype=complex)
    rAvs = np.zeros(nb_points, dtype=complex)
    tAhs = np.zeros(nb_points, dtype=complex)
    tAvs = np.zeros(nb_points, dtype=complex)
    rBhs = np.zeros(nb_points, dtype=complex)
    rBvs = np.zeros(nb_points, dtype=complex)
    tBhs = np.zeros(nb_points, dtype=complex)
    tBvs = np.zeros(nb_points, dtype=complex)

    n1 = 1
    n2 = 1.5
    Z1 = Z0 / n1  # True for dielectrics only.
    Z2 = Z0 / n2  # True for dielectrics only.

    for i, f in enumerate(fs):
        rh, rv, th, tv = setup(n1, n2, f)
        # P = EH   and   E=ZH   =>   P = E**2 / Z.
        rAhs[i] = np.linalg.norm(rh) ** 2 / Z1
        rAvs[i] = np.linalg.norm(rv) ** 2 / Z1
        tAhs[i] = np.linalg.norm(th) ** 2 / Z2
        tAvs[i] = np.linalg.norm(tv) ** 2 / Z2
        #
        rh, rv, th, tv = setup(n2, n1, f)
        rBhs[i] = np.linalg.norm(rh) ** 2 / Z2
        rBvs[i] = np.linalg.norm(rv) ** 2 / Z2
        tBhs[i] = np.linalg.norm(th) ** 2 / Z1
        tBvs[i] = np.linalg.norm(tv) ** 2 / Z1

    crit_b = np.degrees(np.arcsin(n1.real / n2.real))
    brew_a = np.degrees(np.arctan(n2.real / n1.real))
    brew_b = np.degrees(np.arctan(n1.real / n2.real))

    xs = np.degrees(fs)

    plt.subplot(3, 2, 1)
    plt.axvline(brew_a, linestyle=':', color=style.COLORS_STD[0], label=style.latexT("Brewster"))
    plt.plot(xs, tAhs, '-', color=style.COLORS_STD[1], label=style.latexM("T_\perp"))
    plt.plot(xs, tAvs, '-', color=style.COLORS_STD[2], label=style.latexM("T_\parallel"))
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.title(style.latexM("n_i=1.0, n_t=1.5"), fontsize=10)
    plt.xlim((0, 90))
    plt.ylim((-.1, 1.1))
    plt.legend(loc='center left', prop={"size":6})

    plt.subplot(3, 2, 2)
    plt.axvline(crit_b, linestyle='--', color=style.COLORS_STD[0], label=style.latexT("Critical"))
    plt.axvline(brew_b, linestyle=':', color=style.COLORS_STD[0], label=style.latexT("Brewster"))
    plt.plot(xs, tBhs, '-', color=style.COLORS_STD[1], label=style.latexM("T_\perp"))
    plt.plot(xs, tBvs, '-', color=style.COLORS_STD[2], label=style.latexM("T_\parallel"))
    plt.title(style.latexM("n_i=1.5, n_t=1.0"), fontsize=10)
    plt.xlim((0, 90))
    plt.ylim((-.1, 1.1))
    plt.legend(loc='upper right', prop={"size":6})

    plt.subplot(3, 2, 3)
    plt.axvline(brew_a, linestyle=':', color=style.COLORS_STD[0], label=style.latexT("Brewster"))
    plt.plot(xs, rAhs, '-', color=style.COLORS_STD[1], label=style.latexM("R_\perp"))
    plt.plot(xs, rAvs, '-', color=style.COLORS_STD[2], label=style.latexM("R_\parallel"))
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.xlim((0, 90))
    plt.ylim((-.1, 1.1))
    plt.legend(loc='upper left', prop={"size":6})

    plt.subplot(3, 2, 4)
    plt.axvline(crit_b, linestyle='--', color=style.COLORS_STD[0], label=style.latexT("Critical"))
    plt.axvline(brew_b, linestyle=':', color=style.COLORS_STD[0], label=style.latexT("Brewster"))
    plt.plot(xs, rBhs, '-', color=style.COLORS_STD[1], label=style.latexM("R_\perp"))
    plt.plot(xs, rBvs, '-', color=style.COLORS_STD[2], label=style.latexM("R_\parallel"))
    plt.xlim((0, 90))
    plt.ylim((-.1, 1.1))
    plt.legend(loc='upper right', prop={"size":6})

    plt.subplot(3, 2, 5)
    plt.plot(xs, tAhs + rAhs, '-', color=style.COLORS_STD[1], label=style.latexT("Sum $\perp$"))
    plt.plot(xs, tAvs + rAvs, '--', color=style.COLORS_STD[2], label=style.latexT("Sum $\parallel$"))
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.xlabel(style.latexT("Incidence angle [\\si{\\degree}]"))
    plt.xlim((0, 90))
    plt.ylim((-.1, 1.1))
    plt.legend(loc='center left', prop={"size":6})

    plt.subplot(3, 2, 6)
    plt.axvline(crit_b, linestyle='--', color=style.COLORS_STD[0], label=style.latexT("Critical"))
    plt.plot(xs, tBhs + rBhs, '-', color=style.COLORS_STD[1], label=style.latexT("Sum $\perp$"))
    plt.plot(xs, tBvs + rBvs, '--', color=style.COLORS_STD[2], label=style.latexT("Sum $\parallel$"))
    plt.xlabel(style.latexT("Incidence angle [\\si{\\degree}]"))
    plt.xlim((0, 90))
    plt.ylim((-.1, 1.1))
    plt.legend(loc='upper right', prop={"size":6})

    if USE_PDF:
        plt.savefig("interface_oblique_verification.pdf", bbox_inches='tight')
    else:
        plt.show()



def MakeFigure(width):
    with style.latex(width, scipy.constants.golden * 2 / 3):
        style.pretty()
        plot()

if __name__ == '__main__':
    MakeFigure(style.WIDTH_ARTICLE)


