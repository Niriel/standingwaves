"""
.. module:: sw.networks.grid
   :platform: Unix, Windows
   :synopsis: Wire grid polarizers.

.. moduleauthor:: Bertrand Delforge <b.delforge@sron.nl>

"""
from __future__ import division
import numpy as np
import scipy.constants

c0 = scipy.constants.c
mu0 = scipy.constants.mu_0
# Euler constant.  I couldn't find it in numpy's library.
# Insane amount of digits taken from wikipedia.
euler = 0.57721566490153286060651209008240243104215933593992

def RT(a, d, s, f, propdir):
    kx, ky, kz = propdir
    l = c0 / f  # Wavelength (lambda).
    k = (2 * np.pi) / l  # Wave number.
    w = (2 * np.pi) * f  # Pulsation (omega).
    Z0 = c0 * mu0
    Zs = (1 + 1j) * (mu0 * w / (2 * s))

    # Terms that appear several times in the equations.
    Z = Zs / Z0  # For convenience because that ratio shows up a lot.
    lpdz = l / (np.pi * d * kz)
    lnd2pa = np.log(d / (2 * np.pi * a))  # ln.
    kx21 = (1 - kx ** 2)
    k1 = k * kx21 ** .5  # Called k' in the article.
    ka2 = k * a / 2
    k1a2 = k1 * a / 2

    Nx = complex(1, Z * ka2)
    Nc = complex(1, Z / ka2)

    Dx = kx21 * (
                    (
                        lpdz -
                        k1a2 ** 2 +
                        (kx21 * Z0 * s * l * np.pi) ** (-.5) / k1a2
                    )
                    + 2j / np.pi *
                    (
                        lnd2pa +
                        np.pi ** 2 / 6 * (d * kz / l) ** 2 -
                        k1a2 ** 2 * (1 - euler - np.log(k1a2)) +
                        (kx21 * Z0 * s * l * 4 / np.pi) ** (-.5) / k1a2
                    )
                )

    Dc = -kx21 * Z * (2 / np.pi) * lnd2pa + \
             1j * (l / (np.pi ** 2 * a) + kx21 * Z * lpdz)

    common_x = Nx / Dx * l / (np.pi * d)
    common_c = Nc / Dc * a / d
    Rxx = -common_x * (1 - kx * kx) / kz
    Rxy = 0
    Rxz = 0
    Ryx = +common_x * kx * ky / kz
    Ryy = +common_c * kz
    Ryz = -common_c * ky
    Rzx = -common_x * kx
    Rzy = +common_c * ky
    Rzz = -common_c * ky * ky / kz
    Txx = 1 + Rxx
    Txy = 0
    Txz = 0
    Tyx = +common_c * kx * ky / kz
    Tyy = -common_c * kz + 1
    Tyz = +common_c * ky
    Tzx = +common_x * kx
    Tzy = +common_c * ky
    Tzz = -common_c * ky * ky / kz + 1
    R = np.array([[Rxx, Rxy, Rxz],
                      [Ryx, Ryy, Ryz],
                      [Rzx, Rzy, Rzz]])
    T = np.array([[Txx, Txy, Txz],
                      [Tyx, Tyy, Tyz],
                      [Tzx, Tzy, Tzz]])
    return R, T
