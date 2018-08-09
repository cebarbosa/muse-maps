# -*- coding: utf-8 -*-
""" 

Created on 07/08/18

Author : Carlos Eduardo Barbosa

Program to run the modeling of stellar populations.

"""
from __future__ import print_function, division, absolute_import

import os
import sys
import pickle

import numpy as np
from astropy.io import fits
from astropy.table import Table
import pymc3 as pm

import context
from bsf.bsf.bsf import BSF

def fit(idx, redo=False, parametric=False):
    """ Perform the fitting in one of the spectra. """
    outfolder = "pfit" if parametric else "npfit"
    home_dir = os.path.join(context.home, "ssp_modeling",
                            "hydraimf_w4700_9100_dw10_sigma350_test_ssps")
    # Reading spectrum
    data_dir = os.path.join(home_dir, "data")
    filenames = sorted(os.listdir(data_dir))
    filename = filenames[int(idx)]
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
    results_dir = os.path.join(home_dir, outfolder)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    dbname = os.path.join(results_dir, filename.replace(".fits", ".db"))
    summary = os.path.join(results_dir, filename.replace(".fits", ".txt"))
    data = Table.read(os.path.join(data_dir, filename))
    flux = data["flux"]
    obswave = data["obswave"]
    if os.path.exists(dbname) and not redo:
        return
    bsf = BSF(obswave, flux, templates, params=params)
    if parametric:
        bsf.build_parametric_model()
    else:
        bsf.build_nonparametric_model()
    bsf.NUTS_sampling()

    # if parametric:
    #     csp = bsf.PFit(obswave, flux, templates, params)
    #     pass
    # else:
    #     csp = bsf.NPFit(obswave, flux, templates, reddening=True)
    #     csp.NUTS_sampling(nsamp=5, sample_kwargs={"tune": 5})
    #     data = {'model': csp.model, 'trace': csp.trace}
    #     with open(dbname, 'wb') as f:
    #         pickle.dump(data, f)
    #     df = pm.df_summary(csp.trace)
    #     df.to_csv(summary)


if __name__ == "__main__":
    # Append job number for testing purposes
    if len(sys.argv) == 1:
        sys.argv.append("2")
    fit(sys.argv[1], parametric=True)