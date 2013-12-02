from __future__ import division
USE_PDF = False
import numpy as np
import matplotlib
if USE_PDF:
    matplotlib.use('pdf')
import scipy.constants
import matplotlib.pyplot as plt
import style.my_style as style

def plot():
    plt.clf()
    f = 500e9
    omega = 2 * np.pi * f
    n = 1
    c = scipy.constants.speed_of_light / n
    lambda_ = c / f
    k = 2 * np.pi / lambda_
    phi = 0
    x = np.linspace(0, 1e-3, 1000)
    ts = np.linspace(0, 1 / f / 2, 5)
    legends = []
    for i, t in enumerate(ts):
        y = np.cos(omega * t - k * x + phi)
        j = len(ts) - 1 - i
        col = j / (j + 1)
        plt.plot(x * 1000, y, color=(col, col, col))
        plt.ylim([-1.1, 1.1])
        plt.xlabel(style.latexT("Position $z$ [\\si{\\milli\\meter}]"))
        plt.ylabel(style.latexT("Amplitude $e(z, t)$ [\\si{\\volt\\per\\meter}]"))
        legend = style.latexM("t=\\SI{%04.2f}{\\pico\\second}" % (t * 1e12))
        legends.append(legend)
    plt.legend(legends, loc="lower right")
    if USE_PDF:
        plt.savefig("plane_wave_propagation.pdf", bbox_inches='tight')
    else:
        plt.show()


def MakeFigure(width):
    with style.latex(width):
        style.pretty()
        plot()

if __name__ == '__main__':
    MakeFigure(style.WIDTH_ARTICLE)
