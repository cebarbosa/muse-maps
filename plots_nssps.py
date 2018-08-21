# -*- coding: utf-8 -*-
""" 

Created on 17/08/18

Author : Carlos Eduardo Barbosa

Plots for the analysis of the traces obtained with BSF / NSSPs.

"""
from __future__ import print_function, division, absolute_import

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
import pymc3 as pm

import context
from bsf.bsf.bsf import BSF

def plot_fitting(bsf):
    """ Plot the best fit results in comparison with the data. """
    w = bsf.trace["w"]
    wp = bsf.trace["mpoly"]
    f0 = bsf.trace["f0"]
    idxs = []
    for i, p in enumerate(bsf.params.colnames):
        idxs.append(bsf.trace["{}_idx".format(p)])
    idxs = np.array(idxs)
    nchains = idxs.shape[1]
    bestfits = np.zeros((nchains, len(bsf.wave)))
    mpoly = np.zeros_like(bestfits)
    for i in np.arange(nchains):
        idx = idxs[:, i, :]
        ssps =  bsf.templates[idx[0], idx[1], idx[2], idx[3], idx[4], :]
        mpoly[i] = f0[i] * np.dot(wp[i], bsf.mpoly)
        bestfits[i] = np.dot(w[i], ssps) *  mpoly[i]
    bestfit = np.percentile(bestfits, 50, axis=0)
    p95 = np.percentile(bestfits, 95, axis=0)
    p05 = np.percentile(bestfits, 5, axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
    ax1.plot(bsf.wave, bsf.flux, "-")
    ax1.plot(bsf.wave, bestfit, "-")
    ax1.plot(bsf.wave, np.percentile(mpoly, 50, axis=0), "-",
                                     c="C2")
    ax1.plot(bsf.wave, np.percentile(mpoly, 5, axis=0), "-", c="C2")
    ax1.plot(bsf.wave, np.percentile(mpoly, 95, axis=0), "-", c="C2")
    ax1.plot(bsf.wave, p95, "-", c="C1")
    ax1.plot(bsf.wave, p05, "-", c="C1")
    ax1.plot
    ax2.plot(bsf.wave, (bsf.flux - bestfit) / bsf.flux)
    plt.show()
    return


if __name__ == "__main__":
    # style_list = ['default', 'classic'] + sorted(
    #     style for style in plt.style.available if style != 'classic')
    # print(style_list)
    redo = False
    home_dir = os.path.join(context.home, "ssp_modeling",
                            "hydraimf_w4700_9100_dw2_sigma350_all_ssps")
    statmodel = "nssps"
    data_dir = os.path.join(home_dir, "data")
    results_dir = os.path.join(home_dir, statmodel)
    # Creating directories to save the plots
    plots_dir = os.path.join(home_dir, "plots")
    corner_dir = os.path.join(plots_dir, "corner")
    bestfit_dir = os.path.join(plots_dir, "bestfit")
    for directory in [plots_dir, corner_dir, bestfit_dir]:
        if not os.path.exists(directory):
            os.mkdir(directory)
    filenames = sorted(os.listdir(data_dir))
    ###########################################################################
    # Loading templates
    templates_file = os.path.join(home_dir, "templates",
                                  "emiles_templates.fits")
    templates = fits.getdata(templates_file, 0)
    templates = np.array(templates, dtype=np.float)
    wave = fits.getdata(templates_file, 1)
    params = fits.getdata(templates_file, 2)
    params = Table(params)
    templates_norm = params["norm"]
    del params["norm"]
    params.rename_column("[Z/H]", "Z")
    params.rename_column("age", "logT")
    params.rename_column("[alpha/Fe]", "E")
    params.rename_column("[Na/Fe]", "Na")
    # params["logT"] = np.log10(params["logT"])
    ###########################################################################
    for filename in filenames:
        print(filename)
        dbname = os.path.join(results_dir, filename.replace(".fits", "_db"))
        if not os.path.exists(dbname):
            continue
        cornerout = os.path.join(corner_dir, "{}.png".format(
                                 filename.replace(".fits", "")))
        bestfitout = os.path.join(bestfit_dir, "{}.png".format(
                                 filename.replace(".fits", "")))
        data = Table.read(os.path.join(data_dir, filename))
        flux = data["flux"]
        obswave = data["obswave"]
        bsf = BSF(obswave, flux, templates, params=params, statmodel=statmodel)

        with bsf.model:
            bsf.trace = pm.backends.text.load(dbname)
        with plt.style.context("seaborn-paper"):
            plt.rcParams["text.usetex"] = True
            if not os.path.exists(cornerout) or redo:
                bsf.plot_corner(labels=[r"$\alpha - 1$", "[Z/H]", "Age (Gyr)",
                                        r"[$\alpha$/Fe]", "[Na/Fe]"])
                plt.savefig(cornerout, dpi=300)
                plt.show()
                plt.close()
            if not os.path.exists(bestfitout):
                plot_fitting(bsf)
                plt.savefig(cornerout, dpi=300)
                plt.show()
            plt.clf()