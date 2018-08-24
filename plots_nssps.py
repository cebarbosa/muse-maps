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

def plot_fitting(bsf, dbname):
    """ Plot the best fit results in comparison with the data. """
    with bsf.model:
        # wp = np.array([bsf.trace["mpoly_{}".format(i)] for i in range(
        #                bsf.mdegree + 1)]).T
        wp = bsf.trace["mpoly"]
        w = bsf.trace["w"]
        f0 = bsf.trace["f0"]
    N = bsf.Nssps
    idxs = []
    for i, p in enumerate(bsf.params.colnames):
        idxs.append(bsf.trace["{}_idx".format(p)])
    idxs = np.array(idxs)
    nchains = idxs.shape[1]
    bestfits = np.zeros((nchains, len(bsf.wave)))
    mpoly = np.zeros_like(bestfits)
    csps = np.zeros_like(bestfits)
    for i in np.arange(nchains):
        idx = idxs[:, i, :]
        ssps =  bsf.templates[idx[0], idx[1], idx[2], idx[3], idx[4], :]
        csps[i] = np.dot(w[i].T, ssps)
        mpoly[i] = np.dot(wp[i].T, bsf.mpoly)
        bestfits[i] = mpoly[i] *  csps[i]
    bestfit = np.mean(bestfits, axis=0)
    p95 = np.percentile(bestfits, 95, axis=0)
    p05 = np.percentile(bestfits, 5, axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1, gridspec_kw = {'height_ratios':[3, 1]})
    for ax in (ax1, ax2):
        ax.tick_params(right=True, top=True, axis="both",
                       direction='in', which="both")
        ax.minorticks_on()
    ax1.plot(bsf.wave, bsf.flux, "-", c="C0")
    ax1.plot(bsf.wave, bestfit, "-", c="C1")
    # ax1.plot(bsf.wave, np.percentile(mpoly, 50, axis=0), "-",
    #                                  c="C2")
    # ax1.plot(bsf.wave, np.percentile(mpoly, 5, axis=0), "-", c="C2")
    # ax1.plot(bsf.wave, np.percentile(mpoly, 95, axis=0), "-", c="C2")
    ax1.plot(bsf.wave, p95, "-", c="C1")
    ax1.plot(bsf.wave, p05, "-", c="C1")
    ax2.plot(bsf.wave, (bsf.flux - bestfit) / bsf.flux)
    # plt.show()
    return

def run(redo=False):
    # style_list = ['default', 'classic'] + sorted(
    #     style for style in plt.style.available if style != 'classic')
    # print(style_list)
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
        print(dbname)
        if not os.path.exists(dbname):
            continue
        cornerout = os.path.join(corner_dir, "{}.png".format(
            filename.replace(".fits", "")))
        bestfitout = os.path.join(bestfit_dir, "{}.png".format(
            filename.replace(".fits", "")))
        data = Table.read(os.path.join(data_dir, filename))
        flux = data["flux"]
        obswave = data["obswave"]
        bsf = BSF(obswave, flux, templates, params=params, statmodel=statmodel,
                  mdegree=50, reddening=False, Nssps=50)
        print("Loading trace...")
        with bsf.model:
            bsf.trace = pm.backends.text.load(dbname)
        plt.style.context("seaborn-paper")
        plt.rcParams["text.usetex"] = True
        if not os.path.exists(cornerout) or redo:
            print("Producing corner figure...")
            bsf.plot_corner(labels=[r"$\alpha - 1$", "[Z/H]", "Age (Gyr)",
                                    r"[$\alpha$/Fe]", "[Na/Fe]"])
            plt.savefig(cornerout, dpi=300)
            # plt.show()
            plt.close()
        print("Ploting bestfit estimate...")
        plot_fitting(bsf, dbname)
        plt.savefig(bestfitout, dpi=300)
        plt.show()
        plt.clf()

if __name__ == "__main__":
    run()
