# -*- coding: utf-8 -*-
""" 

Created on 11/05/18

Author : Carlos Eduardo Barbosa

Model stellar populations of MUSE data.

"""
from __future__ import print_function, division

import os
import pickle

import numpy as np
import astropy.units as u
from astropy import constants
from scipy.ndimage.filters import gaussian_filter1d
from astropy.table import Table

from spectres import spectres

import context
from run_ppxf import pPXF

def prepare_spectra(outw1, outw2, dw, dataset="MUSE-DEEP", redo=False,
                velscale=None, sigma=350):
    """ Prepare spectra for CSP modeling """
    velscale = context.velscale if velscale is None else velscale
    targetSN = 70
    w1 = 4500
    w2 = 10000
    regrid = np.arange(outw1, outw2, dw)
    for field in context.fields:
        wdir = os.path.join(context.data_dir, dataset, field)
        data_dir = os.path.join(wdir, "ppxf_vel{}_w{}_{}_sn{}".format(int(
            velscale), w1, w2, targetSN))
        pkls = sorted([_ for _ in os.listdir(data_dir) if _.endswith(".pkl")])
        outdir = os.path.join(wdir, "spec1d_resamp_w{}_{}_dw{}".format(outw1,
                              outw2, dw))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        outw1, outw2 = [], []
        for j, pkl in enumerate(pkls):
            print("Working with file {} ({}/{})".format(pkl, j+1, len(pkls)))
            outfile = os.path.join(outdir, pkl.replace(".pkl", ".fits"))
            ####################################################################
            # Reading data from pPXF fitting
            with open(os.path.join(data_dir, pkl)) as f:
                pp = pickle.load(f)
            ####################################################################
            # Subtracting emission lines
            wave = pp.table["wave"]
            flux = pp.table["flux"] - pp.table["emission"]
            # Convolve spectrum to given sigma
            losvd = pp.sol[0]
            z = losvd[0] * u.km / u.s / constants.c
            if losvd[1] > sigma:
                continue
            sigma_diff = np.sqrt(sigma ** 2 - losvd[1] ** 2) / pp.velscale
            flux = gaussian_filter1d(flux, sigma_diff, mode="constant",
                                     cval=0.0)
            ####################################################################
            # De-redshift and resample of the spectrum
            w0 = wave / (1 + z)
            # Resampling the spectra
            fregrid = spectres(regrid, w0, flux)
            table = Table([regrid, fregrid], names=["wave", "flux"])
            table.write(outfile, format="fits", overwrite=True)
            ####################################################################

if __name__ == "__main__":
    # Parameters for the resampling
    outw1 = 4700
    outw2 = 9200
    dw = 10
    prepare_spectra(outw1, outw2, dw, redo=False)