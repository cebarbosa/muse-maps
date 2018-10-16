# -*- coding: utf-8 -*-
"""

Created on 01/07/16

@author: Carlos Eduardo Barbosa

Implementation of known models for dust extinction

"""

import numpy as np
import matplotlib.pyplot as plt

from astropy import units as u

def k_CCM89(w, Rv=3.1):
    """ Extinction curve from Cardelli, Clayton and Mathis 1989.

    Values are assumed to be in Angstrom if units are not given.

    """
    w = np.atleast_1d(w)
    if not hasattr(w, "_unit"):
        w = w * u.angstrom
    k_lambda = np.zeros(len(w))
    x = 1 / w.to("micrometer") * u.micrometer
    # The extinction curve is separated into three segments
    # Infrared section
    idxir = np.where((0.3 <= x) & (x <= 1.1))
    if idxir:
        k_lambda[idxir] =  0.574 * np.power(x[idxir], 1.61) \
                        - 0.527 * np.power(x[idxir], 1.61) / Rv
    # Optical and NIR
    idxop = np.where((1.1 <= x) & (x <= 3.3))
    if idxop:
        y = x[idxop] - 1.82
        ax = 1 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3 + \
             0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6 + 0.32999 * y**7
        bx = 1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3 - 5.38434 * y**4 \
             -0.62251 * y**5 + 5.30260 * y**6 - 2.09002 * y**7
        k_lambda[idxop] = ax + bx / Rv
    # Ultraviolet and Far -UV
    idxuv = np.where((3.3 <= x) & (x <= 8))
    if idxuv:
        Fa = np.zeros(len(idxuv[0]))
        Fb = np.zeros_like(Fa)
        idxF = np.where((5.9 <= x[idxuv]) & (x[idxuv] <= 8))
        Fa[idxF] = -0.04473 * np.power(x[idxF] - 5.9, 2) \
                   - 0.009779 * np.power(x[idxF] - 5.9, 3)
        Fb[idxF] = 0.2130 * np.power(x[idxF] - 5.9, 2) \
                   + 0.1207 * np.power(x[idxF] - 5.9, 3)
        ax = 1.752 - 0.316 * x[idxuv] - 0.104 / (np.power(x[idxuv] - \
                                                 4.67, 2) + 0.341) + Fa
        bx = -3.090 + 1.825 * x[idxuv] + 1.206 / (np.power(x[idxuv] - \
                                                  4.6, 2) + 0.263) + Fb
        k_lambda[idxuv] = ax + bx / Rv
    return k_lambda

def k_C00(w, Rv=3.1):
    """ Extinction curve from Calzetti et al. 2000

    Values are assumed to be in Angstrom if units are not given.

    """
    w = np.atleast_1d(w)
    if not hasattr(w, "_unit"):
        w  = w * u.angstrom
    elif w.unit == u.dimensionless_unscaled:
        w  = w * u.angstrom
    x = 1 / w.to("micrometer") * u.micrometer
    return np.where(w > 0.63 * u.micrometer, 2.659 * (-1.857 + 1.040 * x) + Rv,
                2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 +0.011*x**3) + Rv)


def test_CCM89():
    """ Test the extinction functions with data in table 3 from CCM89. """
    wave = np.linspace(1300, 32000, 5000) * u.angstrom
    ccm89 = k_CCM89(wave)
    c00 = k_C00(wave) / k_C00(5461)
    x = np.array([2.78, 2.27, 1.82, 1.43, 1.11, 0.8, 0.63, 0.46, 0.26])
    y = np.array([1.569, 1.337, 1., 0.751, 0.479, 0.282, 0.190,
                          0.114, 0.056])
    plt.plot(1 / wave.to("micrometer"), ccm89, "-r", label="CCM89")
    plt.plot(1 / wave.to("micrometer"), c00, "-g",
             label="Calzetti et al. 2000")
    plt.plot(x, y, "or", label="CCM98 (Table 3)")
    plt.axhline(y=1, ls="--", c="k")
    plt.legend(loc=2)
    # plt.xlim(0, 3)
    # plt.ylim(0, 2)
    plt.ylabel("$A_\lambda / A_V$")
    plt.xlabel("$1/\lambda$ ($\mu$m$^{-1}$)")
    plt.minorticks_on()
    plt.show()

def calc_balmer():
    """ Calculate the expected correction from Balmer decrement. """
    w_ha = 6562.819
    w_hb = 4861.333
    w_V = 5464
    print -2.5 / (k_C00(w_ha) - k_C00(w_hb))

if __name__ == "__main__":
    test_CCM89()
    # calc_balmer()