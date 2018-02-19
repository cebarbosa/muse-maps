# -*- coding: utf-8 -*-
"""

Created on 04/05/16

@author: Carlos Eduardo Barbosa

Calculates and plot the spectral resolution of MUSE.

"""
from __future__ import print_function
import os

import numpy as np
from astropy import units as u
from astropy.io import fits
from scipy.interpolate import interp1d
from scipy.ndimage.filters import gaussian_filter1d
from specutils.io import read_fits

import context

def get_muse_fwhm():
    """ Returns the FWHM of the MUSE spectrograph as a function of the
    wavelength. """
    wave, R = np.loadtxt(os.path.join(os.path.dirname(
        os.path.abspath(__file__)), "tables/muse_wave_R.dat")).T
    wave = wave *u.nm
    fwhm = wave.to("angstrom") / R
    # First interpolation to obtain extrapolated values
    f1 = interp1d(wave.to("angstrom"), fwhm, kind="linear", bounds_error=False,
                 fill_value="extrapolate")
    # Second interpolation using spline
    wave = np.hstack((4000, wave.to("angstrom").value, 10000))
    f = interp1d(wave, f1(wave), kind="cubic", bounds_error=False)
    return f

def broad2res(w, specs, obsres, res=2.95):
    """ Broad resolution of observed spectra to a given resolution.

    Input Parameters
    ----------------
    w : np.array
        Wavelength array

    specs: One or more spectra to be broadened to the desired resolution.

    obsres : float or np.array
        Observed wavelength spectral resolution FWHM.

    res: float
        Resolution FWHM  of the spectra after the broadening.

    Output parameters
    -----------------
    np.array:
        Broadened spectra.

    """
    specs = np.atleast_2d(specs)
    dw = np.diff(w)[0]
    sigma_diff = np.sqrt(res**2 - obsres**2) / 2.3548 / dw
    broad = np.zeros_like(specs)
    # print "Processing broadening"
    for i,spec in enumerate(specs):
        # print "Spectra {0}/{1}".format(i+1, len(specs))
        d = np.diag(spec)
        for j in range(len(w)):
            d[j] = gaussian_filter1d(d[j], sigma_diff[j], mode="constant",
                                     cval=0.0)
        broad[i] = d.sum(axis=0)
    return broad

def plot_muse_fwhm():
    f = get_muse_fwhm()
    wave = np.linspace(4000, 10000, 1000)
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-paper")
    plt.figure(1)
    plt.minorticks_on()
    plt.plot(wave, f(wave), "-")
    plt.xlabel("$\lambda$ ($\AA$)")
    plt.ylabel(r"Spectral resolution $\alpha$ FWHM (Angstrom)")
    plt.show()

def plot_vel_resolution():
    f = get_muse_fwhm()
    wave = np.linspace(4000, 10000, 1000)
    from astropy.constants import c
    import matplotlib.pyplot as plt
    plt.style.use("seaborn-paper")
    plt.figure(1)
    plt.minorticks_on()
    plt.plot(wave, c.to("km/s") * f(wave) / wave / 2.634, "-")
    plt.xlabel("$\lambda$ ($\AA$)")
    plt.ylabel(r"Velocity scale - sigma (km/s)")
    plt.show()

def broad_binned(fields, res, targetSN=70, dataset="MUSE-DEEP"):
    """ Performs convolution to homogeneize the resolution. """
    for field in fields:
        print(field)
        input_dir = os.path.join(context.data_dir, dataset, field, "spec1d")
        output_dir = os.path.join(context.data_dir, dataset, field,
                                  "spec1d_FWHM{}".format(res))
        if not(os.path.exists(output_dir)):
            os.mkdir(output_dir)
        specs = sorted([_ for _ in os.listdir(input_dir) if "sn{}".format(
                        targetSN) in _])
        for i, filename in enumerate(specs):
            print("Convolving file {} ({} / {})".format(filename, i+1,
                                                        len(specs)))
            filepath = os.path.join(input_dir, filename)
            spec = read_fits.read_fits_spectrum1d(filepath)
            wave = spec.wavelength.to("AA").value
            flux = spec.flux
            muse_fwhm = get_muse_fwhm()
            obsres = muse_fwhm(wave)
            broad = broad2res(wave, flux, obsres, res)[0]
            h = fits.getheader(filepath)
            hdu = fits.PrimaryHDU(data=broad.value, header=h)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(os.path.join(output_dir, filename), overwrite=True)

if __name__ == "__main__":
    # plot_muse_fwhm()
    # plot_vel_resolution()
    broad_binned(context.fields, 2.95)