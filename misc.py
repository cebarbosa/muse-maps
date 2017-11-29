# -*- coding: utf-8 -*-
""" 

Created on 29/11/17

Author : Carlos Eduardo Barbosa

Miscelaneous routines

"""
import numpy as np
from astropy.io import fits

def array_from_header(filename, axis=3, extension=1):
    """ Produces array for wavelenght of a given array. """
    w0 = fits.getval(filename, "CRVAL{0}".format(axis), extension)
    deltaw = fits.getval(filename, "CD{0}_{0}".format(axis), extension)
    pix0 = fits.getval(filename, "CRPIX{0}".format(axis), extension)
    npix = fits.getval(filename, "NAXIS{0}".format(axis), extension)
    return w0 + deltaw * (np.arange(npix) + 1 - pix0)
