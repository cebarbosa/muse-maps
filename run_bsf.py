# -*- coding: utf-8 -*-
""" 

Created on 07/08/18

Author : Carlos Eduardo Barbosa

Program to run the modeling of stellar populations.

"""
from __future__ import print_function, division, absolute_import

import os
import sys

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pymc3 as pm

import context
from bsf.bsf.bsf import BSF

def fit(idx, redo=False, statmodel="nssps"):
    """ Perform the fitting in one of the spectra. """
    home_dir = os.path.join(context.home, "ssp_modeling",
                            "fit_w4700_9100_dw4_sigma350_sn150")
    # Selecting spectrum
    data_dir = os.path.join(home_dir, "data")
    filenames = sorted(os.listdir(data_dir))
    if int(idx) + 1 > len(filenames):
        return
    filename = filenames[int(idx)-1]
    results_dir = os.path.join(home_dir, statmodel)
    dbname = os.path.join(results_dir, filename.replace(".fits", "_db"))
    print(dbname)
    summary = os.path.join(results_dir, filename.replace(".fits",
                                                                ".txt"))
    if os.path.exists(dbname) and not redo:
        return
    # Loading templates
    templates_file= os.path.join(home_dir, "templates", "emiles_templates.fits")
    templates = fits.getdata(templates_file, 0)
    templates = np.array(templates, dtype=np.float)
    wave = fits.getdata(templates_file, 1)
    params = fits.getdata(templates_file, 2)
    params = Table(params)
    norm = params["norm"]
    del params["norm"]
    params.rename_column("[Z/H]", "Z")
    params.rename_column("age", "logT")
    params.rename_column("[alpha/Fe]", "E")
    params.rename_column("[Na/Fe]", "Na")
    params["logT"] = np.log10(params["logT"])
    # Performing fitting
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    data = Table.read(os.path.join(data_dir, filename))
    flux = data["flux"].data
    fluxerr = data["fluxerr"].data
    obswave = data["obswave"]
    bsf = BSF(obswave, flux, templates, params=params,
              reddening=False, mdegree=20, fluxerr=fluxerr)
    if not os.path.exists(dbname) or redo:
        with bsf.model:
            db = pm.backends.Text(dbname)
            bsf.trace = pm.sample(njobs=4, nchains=4, trace=db,
                                  nuts_kwargs={"target_accept": 0.9})
            df = pm.stats.summary(bsf.trace)
            df.to_csv(summary)

if __name__ == "__main__":
    # Append job number for testing purposes
    if len(sys.argv) == 1:
        sys.argv.append("1")
    fit(sys.argv[1], redo=False, statmodel="nssps")