# -*- coding: utf-8 -*-
""" 

Created on 17/08/18

Author : Carlos Eduardo Barbosa

Plots for the analysis of the traces obtained with BSF / NSSPs.

"""
from __future__ import print_function, division, absolute_import

import os
import sys
import pickle

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pymc3 as pm

import context
from bsf.bsf.bsf import BSF

def fit(idx, redo=False, statmodel="nssps"):
    """ Perform the fitting in one of the spectra. """
    # Selecting spectrum
    if os.path.exists(dbname) and not redo:
        return
    if not os.path.exists(dbname) or redo:
        with bsf.model:
            db = pm.backends.Text(dbname)
            bsf.trace = pm.sample(njobs=4, nchains=4, trace=db)
            df = pm.stats.summary(bsf.trace)
            df.to_csv(summary)
    with bsf.model:
        bsf.trace = pm.backends.text.load(dbname)
    bsf.plot()

if __name__ == "__main__":
    style_list = ['default', 'classic'] + sorted(
        style for style in plt.style.available if style != 'classic')
    print(style_list)
    home_dir = os.path.join(context.home, "ssp_modeling",
                            "hydraimf_w4700_9100_dw2_sigma350_all_ssps")
    statmodel = "nssps"
    data_dir = os.path.join(home_dir, "data")
    plots_dir = os.path.join(home_dir, "plots")
    if not os.path.exists(plots_dir):
        os.mkdir(plots_dir)
    results_dir = os.path.join(home_dir, statmodel)
    filenames = sorted(os.listdir(data_dir))
    for filename in filenames:
        print(filename)
        dbname = os.path.join(results_dir, filename.replace(".fits", "_db"))
        summary = os.path.join(results_dir, filename.replace(".fits", ".txt"))
        # Loading templates
        templates_file = os.path.join(home_dir, "templates",
                                      "emiles_templates.fits")
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
        flux = data["flux"]
        obswave = data["obswave"]
        bsf = BSF(obswave, flux, templates, params=params, statmodel=statmodel)
        with bsf.model:
            bsf.trace = pm.backends.text.load(dbname)

        with plt.style.context("seaborn-paper"):
            plt.rcParams["text.usetex"] = True
            corner_dir = os.path.join(plots_dir, "corner")
            if not os.path.exists(corner_dir):
                os.mkdir(corner_dir)
            bsf.plot_corner(labels=[r"$\alpha$", "[Z/H]", "log $T$",
                                    r"[$\alpha$/Fe]", "[Na/Fe]"])
            output = os.path.join(corner_dir, "{}.png".format(
                filename.replace(".fits", "")))
            plt.savefig(output, dpi=300)
            plt.show()