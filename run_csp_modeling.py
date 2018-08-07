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

from astropy.io import fits
from astropy.table import Table
import pymc3 as pm
import matplotlib.pyplot as plt

import context
from bsf.bsf.bsf import NPFit

def fit(idx, redo=False):
    """ Perform the fitting in one of the spectra. """
    home_dir = os.path.join(context.home, "ssp_modeling",
                            "hydraimf_w4700_9100_dw10_sigma350_all_ssps")
    # Reading spectrum
    data_dir = os.path.join(home_dir, "data")
    filenames = sorted(os.listdir(data_dir))
    filename = filenames[int(idx)]
    # Loading templates
    templates_file= os.path.join(home_dir, "templates", "emiles_templates.fits")
    templates = fits.getdata(templates_file, 0)
    wave = fits.getdata(templates_file, 1)
    params = fits.getdata(templates_file, 2)
    # Performing fitting
    results_dir = os.path.join(home_dir, "bsf")

    dbname = os.path.join(results_dir, filename.replace(".fits", ".db"))
    summary = os.path.join(results_dir, filename.replace(".fits", ".txt"))
    data = Table.read(os.path.join(data_dir, filename))
    flux = data["flux"]
    obswave = data["obswave"]
    plt.plot(obswave, flux, "-")
    plt.plot(obswave, templates[1000], "-")
    plt.show()
    if not os.path.exists(dbname) or redo:
        csp = NPFit(obswave, flux, templates, reddening=True)
        csp.NUTS_sampling(nsamp=1000, sample_kwargs={"tune": 1000})
        data = {'model': csp.model, 'trace': csp.trace}
        with open(dbname, 'wb') as f:
            pickle.dump(data, f)
        df = pm.df_summary(csp.trace)
        df.to_csv(summary)


if __name__ == "__main__":
    # Append job number for testing purposes
    if len(sys.argv) == 1:
        sys.argv.append("0")
    fit(sys.argv[1])