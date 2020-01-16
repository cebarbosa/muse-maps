# -*- coding: utf-8 -*-
""" 

Created on 28/11/18

Author : Carlos Eduardo Barbosa

Producing corner plots for NGC 3311
"""
from __future__ import print_function, division

import os
import yaml

import numpy as np
from astropy.table import Table
import astropy.units as u
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm
from spectres import spectres

import context

def plot_ppxf_stpop(bin):
    """ Make corner plot of stellar populations. """
    table_file = "{}_weights.fits".format(bin)
    table = Table.read(table_file)
    cols = table.colnames[:-1]
    pars = [np.unique(table[col].data) for col in cols]
    shape = tuple([len(p) for p in pars])
    weights3D = np.zeros(shape)
    weights = table["mass_weight"].data
    weights /= weights.sum()
    for i, alpha in enumerate(pars[0]):
        for j, metal in enumerate(pars[1]):
            for k, age in enumerate(pars[2]):
                for l, alphaFe in enumerate(pars[3]):
                    for m, nafe in enumerate(pars[4]):
                        idx = np.where((age==table["age"]) & \
                                       (metal==table["[Z/H]"]) & \
                                       (alphaFe==table["[alpha/Fe]"]) & \
                                       (alpha==table["alpha"]) & \
                                       (nafe == table["[Na/Fe]"]))[0][0]
                        weights3D[i,j,k,l,m] = weights[idx]
    plt.figure(figsize=(13,10))
    gs = gridspec.GridSpec(5, 5)
    plt.rcParams["xtick.minor.visible"] = False
    plt.rcParams["ytick.minor.visible"] = False
    plt.rcParams["xtick.direction"] = "inout"
    plt.rcParams["ytick.direction"] = "inout"
    cmap = "viridis"
    color = cm.get_cmap(cmap)(0.2)
    tex = [r"$\alpha_1$", "[Z/H]",  "Age (Gyr)", r"[$\alpha$/Fe]", r"[Na/Fe]"]
    labels = dict(zip(cols, tex))
    for i, xcol in enumerate(cols):
        for j, ycol in enumerate(cols):
            if xcol == ycol:
                # Make histogram
                w = np.sum(weights3D, axis=tuple(np.delete(np.arange(5), i)))
                ax = plt.subplot(gs[i,j])
                ax.bar(np.arange(len(pars[i])), height=w, width=0.9,
                       color=color)
                ax.set_xlim(-0.5, len(pars[i]) -0.5)
                if len(pars[i]) >= 7:
                    plt.xticks(ticks=np.arange(len(pars[i]))[::2],
                               labels=pars[i][::2])
                else:
                    plt.xticks(ticks=np.arange(len(pars[i])), labels=pars[i])
                ax.set_xlabel(labels[ycol])
                ax.yaxis.set_label_position("right")
                ax.yaxis.tick_right()
                ax.set_ylabel("fraction")
            if i > j:
                # Make 2D projection
                axis = np.delete(np.arange(5), [i,j])
                w = np.sum(weights3D, axis=tuple(axis))
                ax = plt.subplot(gs[i, j])
                ax.imshow(w.T, origin="bottom", aspect="auto", cmap=cmap,
                          interpolation="spline16")
                for jj in np.arange(len(pars[j])):
                    ax.axvline(jj, ls="--", c="w", lw=0.5)
                for ii in np.arange(len(pars[i])):
                    ax.axhline(ii, ls="--", c="w", lw=0.5)

                plt.yticks(ticks=np.arange(len(pars[i])), labels=pars[i])
                if j == 0:
                    ax.set_ylabel(labels[xcol])
                else:
                    ax.yaxis.set_ticklabels([])
                ax.set_xlabel(labels[ycol])
                if len(pars[j]) >= 7:
                    plt.xticks(ticks=np.arange(len(pars[j]))[::2],
                               labels=pars[j][::2])
                else:
                    plt.xticks(ticks=np.arange(len(pars[j])), labels=pars[j])
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95,
                        hspace=0.05, wspace=0.05)
    plt.savefig("{}_corner.png".format(bin), dpi=250)
    plt.show()
    plt.close()

def plot_bestfit(bin):
    """ Plot best fit. """
    table = Table.read("{}_bestfit.fits".format(bin))
    values = yaml.load(open("{}.yaml".format(bin)))
    wave = table["lam"].data
    norm = values["flux_norm"] * np.power(10., -20) * u.Unit("erg/s/cm/cm/AA")
    flux = table["galaxy"].data * norm
    error = table["noise"].data * norm
    bestfit = table["bestfit"].data * norm
    gas = table["gas_bestfit"].data * norm
    newwave = np.arange(np.ceil(wave[1]), np.floor(wave[-1]))
    plt.rcParams["xtick.minor.visible"] = True
    plt.rcParams["ytick.minor.visible"] = True
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    label = "{0[0]} Bin {0[2]}".format(bin.replace("field", "Field ").split(
        "_"))
    gs = gridspec.GridSpec(3,1)
    ax1 = plt.subplot(gs[:2,:])
    ax1.fill_between(wave, flux + error - gas, flux - error - gas, lw=4,
                     label=label)
    ax1.plot(wave, bestfit - gas, c="C1", ls="-", label="Model")
    ax1.set_ylabel("Flux (erg s$^{{-1}}$ cm$^{{-2}}$ \\r{{A}}$^{{-1}}$)")
    ax1.xaxis.set_ticklabels([])
    ax1.legend()
    ax2 = plt.subplot(gs[2:,:])
    ax2.fill_between(wave, 100 * (flux + error - bestfit) / bestfit,
                     100 * (flux - error - bestfit)/bestfit)
    ax2.set_ylim(-5, 5)
    ax2.set_ylabel("Resid (\%)")
    ax2.set_xlabel("Wavelength (\\r{{A}})")
    ax2.axhline(y=0, ls="--", c="k", lw=1)
    plt.savefig("{}_fit.png".format(bin), dpi=250)
    plt.clf()


if __name__ == "__main__":
    ############################################################################
    # Local configuration for the code
    targetSN = 250
    velscale = context.velscale
    w1 = 4500
    w2 = 10000
    sample = "bsf"
    ############################################################################
    for field in context.fields:
        wdir = os.path.join(context.data_dir, "MUSE/combined", field,
                            "spec1d_FWHM2.95_sn{}".format(targetSN),
            "ppxf_vel{}_w{}_{}_{}".format(int(velscale), w1, w2, sample))
        if not os.path.exists(wdir):
            continue
        os.chdir(wdir)
        bins = sorted([_.replace(".yaml", "") for _ in os.listdir(".") if
                       _.endswith("yaml")])
        for bin in bins:
            print(bin)
            # plot_ppxf_stpop(bin)
            plot_bestfit(bin)
        break