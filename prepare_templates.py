# -*- coding: utf-8 -*-
""" 

Created on 04/12/17

Author : Carlos Eduardo Barbosa

Prepare stellar population model as templates for pPXF and SSP fitting.

"""
from __future__ import print_function, division

import os
from itertools import product
from datetime import datetime

import numpy as np
import astropy.units as u
from astropy.table import Table
from astropy.io import fits
import ppxf.ppxf_util as util
from tqdm import tqdm

import context
from homogeneize_resolution import broad2res
from misc import array_from_header

class EMiles_models():
    """ Class to handle data from the EMILES SSP models. """
    def __init__(self, sample=None, path=None, w1=4500, w2=10000):
        self.path = os.path.join(context.data_dir,
                        "EMILES_BASTI_w{}_{}".format(w1, w2))
        self.sample = "all" if sample is None else sample
        if self.sample not in ["all", "bsf", "salpeter", "minimal",
                               "kinematics", "test"]:
            raise ValueError("EMILES sample not defined: {}".format(
                self.sample))
        self.values = self.Values(self.sample)

    class Values():
        """ Defines the possible values of the model parameters in different
        samples.
        """
        def __init__(self, sample):
            if sample == "all":
                self.exponents = np.array(
                    [0.3, 0.5, 0.8, 1.0, 1.3, 1.5,
                     1.8, 2.0, 2.3, 2.5, 2.8, 3.0,
                     3.3, 3.5])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 27)
                self.alphaFe = np.array([0., 0.2, 0.4])
                self.NaFe = np.array([0., 0.3, 0.6])
            elif sample == "salpeter":
                self.exponents = np.array([1.3])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 14)
                self.alphaFe = np.array([0., 0.2, 0.4])
                self.NaFe = np.array([0., 0.3, 0.6])
            if sample == "kinematics":
                self.exponents = np.array([1.3])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 14)
                self.alphaFe = np.array([0., 0.4])
                self.NaFe = np.array([0.])
            if sample == "test":
                self.exponents = np.array([1.3])
                self.ZH = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                     0.15,  0.26,  0.4])
                self.age = np.linspace(1., 14., 14)
                self.alphaFe = np.array([0])
                self.NaFe = np.array([0.])
            return

    def get_filename(self, imf, metal, age, alpha, na):
        """ Returns the name of files for the EMILES library. """
        msign = "p" if metal >= 0. else "m"
        esign = "p" if alpha >= 0. else "m"
        azero = "0" if age < 10. else ""
        nasign = "p" if na >= 0. else "m"
        return "Ebi{0:.2f}Z{1}{2:.2f}T{3}{4:02.4f}_Afe{5}{6:2.1f}_NaFe{7}{" \
               "8:1.1f}.fits".format(imf, msign, abs(metal), azero, age, esign,
                                     abs(alpha), nasign, na)

def prepare_templates_emiles_muse(velscale=50, sample="all", redo=False,
                                  fwhm=None):
    """ Pipeline for the preparation of the templates."""
    output = os.path.join(context.data_dir, "templates",
            "emiles_vel{}_{}_fwhm{}.fits".format(velscale, sample, fwhm))
    if os.path.exists(output) and not redo:
        return
    fwhm = 2.95 if fwhm is None else fwhm
    emiles = EMiles_models(sample=sample)
    # First part: getting SSP templates
    grid = np.array(np.meshgrid(emiles.values.exponents, emiles.values.ZH,
                             emiles.values.age, emiles.values.alphaFe,
                             emiles.values.NaFe)).T.reshape(-1, 5)

    filenames = []
    for args in grid:
        filenames.append(os.path.join(emiles.path, emiles.get_filename(*args)))
    dim = len(grid)
    params = np.zeros((dim, 5))
    # Using first spectrum to build arrays
    filename = os.path.join(emiles.path, emiles.get_filename(*grid[0]))
    spec = Table.read(filename, format="fits")
    wave = spec["wave"]
    wrange = [wave[0], wave[-1]]
    flux = spec["flux"]
    newflux, logLam, velscale = util.log_rebin(wrange, flux,
                                               velscale=velscale)
    ssps = np.zeros((dim, len(logLam)))
    # Iterate over all models
    newfolder = os.path.join(context.data_dir, "templates", \
                             "vel{}_fwhm{}".format(velscale, fwhm))
    if not os.path.exists(newfolder):
        os.mkdir(newfolder)
    ############################################################################
    # Sub routine to process a single spectrum
    def process_spec(filename, velscale, redo=False):
        outname = os.path.join(newfolder, os.path.split(filename)[1])
        if os.path.exists(outname) and not redo:
            return
        spec = Table.read(filename, format="fits")
        flux = spec["flux"].data
        if fwhm > 2.51:
            flux = broad2res(wave, flux.T, np.ones_like(wave) * 2.51,
                             res=fwhm).T
        newflux, logLam, velscale = util.log_rebin(wrange, flux,
                                               velscale=velscale)
        hdu = fits.PrimaryHDU(newflux)
        hdu.writeto(outname, overwrite=True)
        return
    ############################################################################
    print(" Processing SSP spectra...")
    for i, fname in enumerate(tqdm(filenames)):
        process_spec(fname, velscale)
    for i, args in enumerate(tqdm(grid)):
        filename = os.path.join(newfolder, emiles.get_filename(*args))
        data = fits.getdata(filename)
        ssps[i] = data
        params[i] = args
    hdu1 = fits.PrimaryHDU(ssps)
    hdu1.header["EXTNAME"] = "SSPS"
    params = Table(params, names=["imf", "Z", "T", "alphaFe", "NaFe"])
    remove_cols = []
    for param in params.colnames:
        n = len(np.unique(params[param]))
        if n == 1:
            remove_cols.append(param)
    if len(remove_cols) > 0:
        params.remove_columns(remove_cols)
    hdu3 = fits.BinTableHDU(params)
    hdu3.header["EXTNAME"] = "PARAMS"
    # Making wavelength array
    hdu4 = fits.BinTableHDU(Table([logLam], names=["loglam"]))
    hdu4.header["EXTNAME"] = "LOGLAM"
    hdulist = fits.HDUList([hdu1, hdu3, hdu4])
    hdulist.writeto(output, overwrite=True)
    return

if __name__ == "__main__":
    prepare_templates_emiles_muse(velscale=50, sample="kinematics", fwhm=2.95,
                                  redo=False)