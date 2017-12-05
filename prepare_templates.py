# -*- coding: utf-8 -*-
""" 

Created on 04/12/17

Author : Carlos Eduardo Barbosa

Prepare stellar population model as templates for pPXF and SSP fitting.

"""
from __future__ import print_function, division

import os
from itertools import product

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table

from specutils.io import read_fits

import context

class EMiles_models():
    """ Class to handle data from the EMILES SSP models. """
    def __init__(self, sample=None):
        self.path = os.path.join(context.home,
                                 "models/EMILES_BASTI_INTERPOLATED")
        self.sample = "all" if sample is None else sample
        if self.sample not in ["all", "test"]:
            raise(ValueError, "Subsample not valid")
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
                                               0.15,
                                               0.26, 0.4])
                self.age = np.linspace(1., 14., 27)
                self.alphaFe = np.array([0., 0.2, 0.4])
                self.NaFe = np.array([0., 0.3, 0.6])
            elif sample == "testing":
                self.exponents = np.array([2.3, 2.5])
                self.ZH = np.array([0.06, 0.15])
                self.age = np.array([10., 14.])
                self.alphaFe = np.array([0., 0.2])
                self.NaFe = np.array([0., 0.3])
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

def trim_templates(emiles, w1=4500, w2=10000):
    """ Slice spectra from templates according to wavelength range. """
    newpath = os.path.join(context.home, "models/EMILES_BASTI_w{}_{}".format(
                           w1, w2))
    if not os.path.exists(newpath):
        os.mkdir(newpath)
    for args in product(emiles.values.exponents, emiles.values.ZH,
                        emiles.values.age, emiles.values.alphaFe,
                        emiles.values.NaFe):
        filename = os.path.join(emiles.path, emiles.get_filename(*args))
        newfilename = os.path.join(newpath, emiles.get_filename(*args))
        if os.path.exists(newfilename):
            continue
        spec = read_fits.read_fits_spectrum1d(filename)
        idx = np.where(np.logical_and(spec.dispersion > w1, spec.dispersion
                                      < w2))
        tab = Table([spec.dispersion[idx] * u.AA, spec.flux[idx] * u.adu],
                    names=["wave", "flux"])
        tab.write(newfilename, format="fits")
        print("Created file ", newfilename)

    return



if __name__ == "__main__":
    emiles = EMiles_models()
    trim_templates(emiles)