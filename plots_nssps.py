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

def plot_fitting(bsf, norm=1., spec=None):
    """ Plot the best fit results in comparison with the data. """
    spec = "Data" if spec is None else spec
    with bsf.model:
        # wp = np.array([bsf.trace["mpoly_{}".format(i)] for i in range(
        #                bsf.mdegree + 1)]).T
        wp = bsf.trace["mpoly"]
        w = bsf.trace["w"]
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
    bestfit = np.percentile(bestfits, 50, axis=0)
    percupper = np.percentile(bestfits, 50 + 34.14, axis=0)
    perclower = np.percentile(bestfits, 50 - 34.14, axis=0)
    fig, (ax1, ax2) = plt.subplots(2, 1,
                                   gridspec_kw = {'height_ratios':[3, 1]},
                                   figsize=(3.32153, 2.5))
    for ax in (ax1, ax2):
        ax.tick_params(right=True, top=True, axis="both",
                       direction='in', which="both")
        ax.minorticks_on()
    ax1.fill_between(bsf.wave, norm * (bsf.flux + bsf.fluxerr),
                     norm * (bsf.flux - bsf.fluxerr), linestyle="-",
                     color="C0", label=spec)
    ax1.fill_between(bsf.wave, norm * perclower, norm * percupper,
                     linestyle="-", color="C1",
                     label="BSF model")
    ax1.set_xticklabels([])
    ax1.set_ylabel("Flux ($10^{-20}$ erg/cm$^{2}$/\\r{A}/s)",
                   fontsize=8)
    ax1.legend(loc=2, prop={'size': 6}, frameon=False)
    ax2.plot(bsf.wave, 100 * (bsf.flux - bestfit) / bsf.flux)
    ax2.axhline(y=0, c="k", ls="--")
    ax2.set_ylim(-5, 5)
    ax2.set_ylabel("Resid. (\%)", fontsize=8)
    ax2.set_xlabel("Rest Wavelength (\\r{A})", fontsize=8)
    plt.subplots_adjust(left=0.14, right=0.985, hspace=0.08, bottom=.16,
                        top=.98)
    return

def run(redo=True):
    # style_list = ['default', 'classic'] + sorted(
    #     style for style in plt.style.available if style != 'classic')
    # print(style_list)
    home_dir = os.path.join(context.home, "ssp_modeling",
                            "fit_w4700_9100_dw2_sigma350_sn150")
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
    mdegrees = [50, 40]
    Nssps = [50, 20]
    for i, filename in enumerate(filenames):
        print(filename)
        name = "Field {0[0][5]} Spec {0[2]}".format(filename[:-5].split("_"))
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
        fluxerr = data["fluxerr"]
        obswave = data["obswave"]
        norm = data["norm"]
        bsf = BSF(obswave, flux, templates, params=params, statmodel=statmodel,
                  mdegree=mdegrees[i], reddening=False, Nssps=Nssps[i],
                  fluxerr=fluxerr)
        print("Loading trace...")
        with bsf.model:
            bsf.trace = pm.backends.text.load(dbname)
        plt.style.context("seaborn-paper")
        plt.rcParams["text.usetex"] = True
        plt.rcParams["font.family"] = "serif"
        plt.rcParams['font.serif'] = 'Computer Modern'
        if not os.path.exists(cornerout) or redo:
            print("Producing corner figure...")
            bsf.plot_corner(labels=[r"$\alpha - 1$", "[Z/H]", "Age (Gyr)",
                                    r"[$\alpha$/Fe]", "[Na/Fe]"])
            plt.savefig(cornerout, dpi=300)
            plt.close()
        print("Ploting bestfit estimate...")
        plot_fitting(bsf, norm=norm, spec=name)
        plt.savefig(bestfitout, dpi=300)
        plt.close()

if __name__ == "__main__":
    run()
