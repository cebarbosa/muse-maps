# -*- coding: utf-8 -*-
""" 

Created on 04/12/17

Author : Carlos Eduardo Barbosa

Prepare stellar population model as templates for pPXF and SSP fitting.

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.io import fits

import context

class EMiles_models():
    def __init__(self, sample=None):
        self.path = os.path.join(context.home,
                                 "models/EMILES_BASTI_INTERPOLATED")
        samples = ["all", "test"]
        self.sample = "all" if sample is None else sample
        self.set_ranges()

    def set_ranges(self):
        self.ranges = {}
        if self.sample == "all":
            self.ranges["exponents"] = np.array([0.3, 0.5,  0.8, 1.0, 1.3, 1.5,
                                                 1.8, 2.0, 2.3, 2.5, 2.8, 3.0,
                                                 3.3, 3.5])
            self.ranges["Z/H"] = np.array([-0.96, -0.66, -0.35, -0.25, 0.06,
                                          0.15,
                                         0.26, 0.4])
            self.ranges["age"] = np.linspace(1., 14., 27)
            self.ranges["alpha/Fe"] = np.array([0., 0.2, 0.4])
            self.ranges["Na/Fe"] = np.array([0., 0.3, 0.6])
        elif self.sample == "testing":
            self.ranges["exponents"] = np.array([2.3, 2.5])
            self.ranges["Z/H"] = np.array([0.06, 0.15])
            self.ranges["age"] = np.array([10., 14.])
            self.ranges["alpha/Fe"] = np.array([0., 0.2])
            self.ranges["Na/Fe"] = np.array([0., 0.3])
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


if __name__ == "__main__":
    emiles = EMiles_models()