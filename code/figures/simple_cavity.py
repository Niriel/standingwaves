from __future__ import division
USE_PDF = True
import numpy as np
import matplotlib
if USE_PDF:
    matplotlib.use('pdf')
import scipy.constants
import matplotlib.pyplot as plt
import style.my_style as style
import sw

def setup(f):
    # f: frequency.
    # Networks.
    n_inside = 1  # Refractive index.
    n_outside = 1.5
    L = .5  # Length of the cavity.
    m0 = sw.networks.InterfaceNormal(n_outside, n_inside)
    space = sw.networks.Distance(n_inside, L, f)
    m1 = sw.networks.InterfaceNormal(n_inside, n_outside)
    # Ports.
    #    0 [m0] 1  -  2 [space] 3  -  4 [m1] 5
    couplings = sw.ExpandCouplingsTo3d(({1, 2}, {3, 4}))
    nb_ports = 6 * 3  # *3 because 3D.
    networks = (m0, space, m1)
    presolve = sw.Solver(nb_ports, couplings)
    solve = presolve(networks)
    # Inputs.
    a = np.zeros(nb_ports, dtype=complex)
    c = np.zeros_like(a)
    a[0:3] = np.array([1, 0, 0])  # x-polarized.
#     c[3:6] = np.array([1, 0, 0])  # x-polarized.
    b = solve(a, c)
    b0 = b[0 * 3:0 * 3 + 3]
    b5 = b[5 * 3:5 * 3 + 3]
    return b0[0], b5[0]

def plot():
    nb_points = 4001
    f_range = 64e9
    f_start = 500e9
    f_stop = f_start + f_range
    fs = np.linspace(f_start, f_stop, nb_points)
    observable = fs <= 504e9
    b0s = np.zeros(nb_points, dtype=complex)
    b5s = np.zeros(nb_points, dtype=complex)
    for i, f in enumerate(fs):
        b0s[i], b5s[i] = setup(f)
    y0s = np.abs(b0s) ** 2
    y5s = np.abs(b5s) ** 2
    # Direct.
    plt.plot(fs[observable] / 1.e9, y0s[observable], color=style.COLORS_STD[1])
    plt.plot(fs[observable] / 1.e9, y5s[observable], color=style.COLORS_STD[2])
    plt.xlabel(style.latexT("Frequency [\\si{\\giga\\hertz}]"))
    plt.ylabel(style.latexT("Power coupling [1]"))
    plt.ylim((-.1, 1.1))
    if USE_PDF:
        plt.savefig("simple_cavity_direct.pdf", bbox_inches='tight')
    else:
        plt.show()
    # FFT.
    L = .5
    prediction = scipy.constants.speed_of_light / (2 * L) / 1.e6
    h = np.hanning(nb_points)
    y0s = np.abs(np.fft.rfft(y0s * h))
    y5s = np.abs(np.fft.rfft(y5s * h))
    cutat = len(y0s)
    p = np.fft.fftfreq(nb_points, f_range / (nb_points))[:cutat]  # In Hz-1.
    p = 1 / p  # In Hz.
    p /= 1.e6  # In MHz.
#     p = p[3:]
#     y0s = y0s[3:]
#     y5s = y5s[3:]
    plt.clf()
#     plt.axvline(prediction / 2, linestyle='-', color='#878686', linewidth=1)
#     plt.axvline(prediction, linestyle='-', color='#878686', linewidth=1)
    plt.plot(p, y0s, '-', color=style.COLORS_STD[1])
    plt.plot(p, y5s, '--', color=style.COLORS_STD[2])
    plt.xlabel(style.latexT("Period [\\si{\\mega\\hertz}]"))
    plt.ylabel(style.latexT("FFT power coupling [arbitrary]"))
    plt.annotate(style.latexM("c/2d"),
                 xy=(prediction, 0),
                 xytext=(prediction, -5),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.annotate(style.latexM("c/4d"),
                 xy=(prediction / 2, 0),
                 xytext=(prediction / 2, -5),
                 horizontalalignment='center',
                 verticalalignment='center')
    plt.xlim((0, 500))
    plt.ylim((-10, 70))
    if USE_PDF:
        plt.savefig("simple_cavity_fft.pdf", bbox_inches='tight')
    else:
        plt.show()



def MakeFigure(width):
    with style.latex(width):
        style.pretty()
        plot()

if __name__ == '__main__':
    MakeFigure(style.WIDTH_ARTICLE)
