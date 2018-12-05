# -*- coding: utf-8 -*-
""" 

Created on 11/05/18

Author : Carlos Eduardo Barbosa

Model stellar populations of MUSE data.

"""
from __future__ import print_function, division

import os
import pickle
import yaml

import numpy as np
import astropy.units as u
from astropy import constants
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from astropy.table import Table, hstack

from spectres import spectres

import context
from misc import array_from_header

def prepare_spectra(table, wfit, outdir):
    """ Prepare spectra for CSP modeling """
    sigma_out = 350
    data = Table.read(table)
    flux = data["galaxy"].data - data["gas_bestfit"].data # sky subtraction
    fluxerr = data["noise"].data
    wave = data["lam"].data
    pars = yaml.load(open(table.replace("_bestfit.fits", ".yaml")))
    v = pars["V_0"]
    sigma = pars["sigma_0"]
    # Convolve spectrum to given sigma
    z = v * u.km / u.s / constants.c
    if sigma > sigma_out:
        return
    sigma_diff = np.sqrt(sigma_out ** 2 - sigma ** 2) / context.velscale
    flux = gaussian_filter1d(flux, sigma_diff, mode="constant", cval=0.0)
    errdiag = np.diag(fluxerr)
    for j in range(len(wave)):
        errdiag[j] = gaussian_filter1d(errdiag[j] ** 2, sigma_diff,
                                       mode="constant", cval=0.0)
    newfluxerr = np.sqrt(errdiag.sum(axis=0))
    ###########################################################################
    # De-redshift and resample of the spectrum
    w0 = wave / (1 + z)
    # Resampling the spectra
    fresamp = spectres(wfit, w0, flux)
    fresamperr = spectres(wfit, w0, newfluxerr)
    w = wfit * (1 + z)
    outtable = Table([wfit, w, fresamp, fresamperr],
                  names=["wave", "obswave", "flux", "fluxerr"])
    output = os.path.join(outdir,
                          os.path.split(table)[1].replace("_bestfit", ""))
    outtable.write(output, format="fits", overwrite=True)
    ############################################################################

def prepare_templates(outw1, outw2, dw, outdir, sigma=350, redo=False,
                      velscale=None,
                      sample=None):
    """ Resample templates for full spectral fitting. """
    velscale = context.velscale if velscale is None else velscale
    sample = "all" if sample is None else sample
    w1 = 4500
    w2 = 10000
    wnorm = 5635
    dnorm = 40
    tempfile = os.path.join(context.home, "templates",
        "emiles_muse_vel{}_w{}_{}_{}.fits".format(int(velscale), w1, w2,
                                                  sample))
    output = os.path.join(outdir, "emiles_templates.fits")
    if os.path.exists(output) and not redo:
        templates = fits.getdata(output, 0)
        wave = fits.getdata(output, 1)
        params = fits.getdata(output, 2)
        return wave, params, templates
    wave = np.exp(array_from_header(tempfile, axis=1, extension=0))

    ssps = fits.getdata(tempfile, 0)
    params = Table.read(tempfile, hdu=2)
    newwave = np.arange(outw1, outw2, dw)
    idx_norm = np.where(np.logical_and(wave > wnorm - dnorm,
                                       wave < wnorm + dnorm))[0]
    templates = np.zeros((len(ssps), len(newwave)))
    norms = np.zeros(len(ssps))
    for i in np.arange(len(ssps)):
        print("Preparing templates ({}/{})".format(i+1, len(ssps)))
        sigma_pix = sigma / velscale
        flux = gaussian_filter1d(ssps[i], sigma_pix, mode="constant",
                                 cval=0.0)
        newflux = spectres(newwave, wave, flux)
        norm = np.median(flux[idx_norm])
        newflux /= norm
        templates[i] = newflux
        norms[i] = norm
    norms = Table([norms], names=["norm"])
    params = hstack([params, norms])
    hdu1 = fits.PrimaryHDU(templates)
    hdu2 = fits.ImageHDU(newwave)
    hdu3 = fits.BinTableHDU(params)
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(output, overwrite=True)
    return newwave, params, templates

def select_lick_wave(wave):
    """ Select the regions of the spectra containing the Lick indices. """
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bandsz0 = np.loadtxt(bandsfile, usecols=(3, 6))
    idxs = []
    for band in bandsz0:
        idx1 = np.where(wave > band[0])[0]
        idx2 = np.where(wave < band[1])[0]
        idxs.append(np.intersect1d(idx1, idx2))
    idxs = np.unique(np.hstack(idxs))
    return idxs

def plot(obswave, flux, norm, dbname, outw1, outw2, dw, velscale=None,
         sample=None, idxs=None):
    """ Plot results from model. """
    velscale = context.velscale if velscale is None else velscale
    sample = "salpeter_regular" if sample is None else sample
    idxs = np.arange(len(obswave)) if idxs is None else idxs
    pkl = dbname.split("/")[-1].replace(".db", ".pkl")
    field, targetSN, n = pkl.split("_")
    ############################################################################
    # Getting results from pPXF for comparison
    w1 = 4500
    w2 = 10000
    dataset = "MUSE-DEEP"
    wdir = os.path.join(context.data_dir, dataset, field)
    pklfile = os.path.join(wdir, "ppxf_vel{}_w{}_{}_{}".format(int(
        velscale), w1, w2, targetSN), pkl)
    with open(pklfile) as f:
        pp = pickle.load(f)
    table = Table(pp.table)
    pwave = table["wave"]
    pbestfit = table["bestfit"] - table["emission"]
    pbestfit = spectres(obswave.value, np.array(pwave), np.array(pbestfit))
    ############################################################################
    tempfile= os.path.join(context.home, "templates",
             "emiles_sigma_350_w{}_{}_dw{}_{}.fits".format(outw1,
                                                                    outw2,
                                                           dw, sample))
    templates = fits.getdata(tempfile, 0)
    wave = fits.getdata(tempfile, 1)
    params = Table.read(tempfile, 2)
    with open(dbname, 'rb') as buff:
        mcmc = pickle.load(buff)
    trace = mcmc["trace"]
    make_corner_plot(trace, params)
    bestfit = trace["bestfit"].mean(axis=0)
    bf05 = np.percentile(trace["bestfit"], 5, axis=0)
    bf95 = np.percentile(trace["bestfit"], 95, axis=0)
    # Plot
    ax1 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax1.minorticks_on()
    ax1.plot(obswave[idxs], (norm * flux)[idxs], "o-")
    ax1.plot(obswave[idxs], norm * bestfit, "o-")
    ax1.fill_between(obswave, norm * bf05, norm * bf95, color="C1", alpha=0.5)
    ax1.plot(obswave, pbestfit)
    ax1.set_xlabel("$\lambda$ (\AA)")
    ax1.set_ylabel("Flux ($10^{-20}$ erg s $^{\\rm -1}$ cm$^{\\rm -2}$ \\r{"
                   "A}$^{\\rm -1}$)")
    ax2 = plt.subplot2grid((3, 1), (2, 0))
    ax2.minorticks_on()
    ax2.plot(obswave, norm * (bestfit - flux), c="C1")
    ax2.fill_between(obswave, norm * (bf05-flux), norm * (bf95 - flux),
                                                          color="C1", alpha=0.5)
    ax2.plot(obswave, pbestfit - norm * flux, c="C2")
    plt.show()

def make_corner_plot(trace, params):
    """ Produces corner plot for relevant variables. """
    weights = trace["w"].mean(axis=0)
    parnames = params.colnames[:-1]
    npars = len(params.colnames[:-1])
    fig = plt.figure(1)
    idxs = np.arange(npars)
    ij = np.array(np.meshgrid(idxs, idxs)).T.reshape(-1, 2)
    widths = [0.1, 0.1, 0.8, 0.15, 0.2]
    for i, j in ij:
        if i == j:
            ax = plt.subplot(npars, npars, j + npars * i + 1)
            ax.minorticks_on()
            values = np.unique(params[parnames[i]])
            w = np.zeros(len(values))
            for k,val in enumerate(values):
                idx = np.where(params[parnames[i]] == val)
                if not len(idx):
                    continue
                w[k] = np.sum(weights[idx] / params["norm"][idx])
                # w[k] = np.sum(weights[idx])
            w /= w.sum()
            ax.bar(values, w, width=widths[i])
        else:
            ax = plt.subplot(npars, npars, j + npars * i + 1)
            ax.minorticks_on()
            v1s = np.unique(params[parnames[i]])
            v2s = np.unique(params[parnames[j]])
            w = np.zeros((len(v1s), len(v2s)))
            for l,v1 in enumerate(v1s):
                idx1 = np.where(params[parnames[i]] == v1)[0]
                for m,v2 in enumerate(v2s):
                    idx2 = np.where(params[parnames[j]] == v2)[0]
                    idx = np.intersect1d(idx1, idx2)
                    w[l,m] = np.sum(weights[idx] / params["norm"][idx])
                    # w[l, m] = np.sum(weights[idx])
            x, y = np.meshgrid(v1s, v2s)
            ax.pcolormesh(y.T, x.T, w, cmap="Greys")
            # ax.contour(y.T, x.T, w, colors="k")
    plt.show()

def prepare_bsf_voronoi(redo=True):
    """ Prepare the modeling of data using Voronoi binning"""
    ############################################################################
    # Input parameters
    w1 = context.w1
    w2 = context.w2
    velscale = context.velscale
    sample = "bsf"
    targetSN = 250
    dataset = "MUSE"
    ############################################################################
    # BSF parameters
    outw1 = 4800
    outw2 = 9100
    dw = 4
    wfit = np.arange(outw1, outw2, dw)
    sigma = 350 # km / s
    outroot = os.path.join(context.data_dir, dataset, "bsf")
    if not os.path.exists(outroot):
        os.mkdir(outroot)
    # Preparing the data
    outdir_data = os.path.join(outroot, "data")
    if not os.path.exists(outdir_data):
        os.mkdir(outdir_data)
    for field in context.fields:
        data_dir = os.path.join(context.data_dir, dataset, "combined", field,
                  "spec1d_FWHM2.95_sn{}".format(targetSN),
                  "ppxf_vel{}_w{}_{}_{}".format(int(velscale), w1, w2, sample))
        if not os.path.exists(data_dir):
            continue
        tables = sorted([_ for _ in os.listdir(data_dir) if _.endswith(
                         "bestfit.fits")])
        for table in tables:
            print("Processing file {}".format(table))
            prepare_spectra(os.path.join(data_dir, table), wfit, outdir_data)
    input(404)

    # Setting unique name for particular modeling
    fitname = "ngc3311_w{}_{}_dw{}_sigma{}_sn{}".format(outw1, outw2, dw, sigma,
                                                    targetSN)
    outroot = os.path.join(context.home, "bsf", fitname)
    if not os.path.exists(outroot):
        os.mkdir(outroot)
    # Setting the directory where the data is going to be saved
    data_dir = os.path.join(outroot, "data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    prepare_spectra(outw1, outw2, dw, data_dir, redo=redo, sigma=sigma,
                    targetSN=targetSN)
    # Setting templates
    templates_dir = os.path.join(outroot, "templates")
    if not os.path.exists(templates_dir):
        os.mkdir(templates_dir)
    wave, params, templates = prepare_templates(outw1, outw2, dw,
                                                templates_dir, redo=redo,
                                                sample=sample, sigma=sigma)

if __name__ == "__main__":
    prepare_bsf_voronoi(redo=True)
