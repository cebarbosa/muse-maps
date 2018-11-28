# -*- coding: utf-8 -*-
""" 

Created on 28/11/18

Author : Carlos Eduardo Barbosa

Producing corner plots for NGC 3311
"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.table import Table
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import cm

import context

def plot_ppxf_stpop(bin):
    """ Make corner plot of stellar populations. """
    table_file = "{}_weights.fits".format(bin)
    table = Table.read(table_file)
    cols = table.colnames[:-1]
    print(cols)
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
    plt.figure(figsize=(12,9))
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
                if len(pars[i]) > 15:
                    plt.xticks(ticks=np.arange(len(pars[i]))[::3],
                               labels=pars[i][::3])
                else:
                    plt.xticks(ticks=np.arange(len(pars[i])), labels=pars[i])
                ax.set_xlabel(labels[ycol])
                ax.set_ylabel("fraction")
            if i > j:
                # Make 2D projection
                axis = np.delete(np.arange(5), [i,j])
                w = np.sum(weights3D, axis=tuple(axis))
                ax = plt.subplot(gs[i, j])
                ax.imshow(w.T, origin="bottom", aspect="auto", cmap=cmap,
                          interpolation="spline16")
                plt.yticks(ticks=np.arange(len(pars[i])), labels=pars[i])
                plt.xticks(ticks=np.arange(len(pars[j])), labels=pars[j])
                ax.set_ylabel(labels[xcol])
                ax.set_xlabel(labels[ycol])
                if len(pars[j]) > 15:
                    plt.xticks(ticks=np.arange(len(pars[j]))[::3],
                               labels=pars[j][::3])
                else:
                    plt.xticks(ticks=np.arange(len(pars[j])), labels=pars[j])
    plt.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)
    plt.savefig("{}_corner.png".format(bin), dpi=250)
    plt.show()
    plt.close()

if __name__ == "__main__":
    ############################################################################
    # Local configuration for the code
    version = 0
    velscale = context.velscale
    library = "miles"
    sample = "bsf"
    ############################################################################
    for field in context.fields:
        wdir = os.path.join(context.data_dir, "MUSE/combined", field,
            "ppxf_ellipv{}_vel{}_{}_{}".format(version, velscale, library,
                                                      sample))
        if not os.path.exists(wdir):
            continue
        os.chdir(wdir)
        bins = sorted([_.replace(".yaml", "") for _ in os.listdir(".") if
                       _.endswith("yaml")])
        for bin in bins:
            plot_ppxf_stpop(bin)