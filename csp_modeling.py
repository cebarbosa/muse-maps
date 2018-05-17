# -*- coding: utf-8 -*-
""" 

Created on 11/05/18

Author : Carlos Eduardo Barbosa

Model stellar populations of MUSE data.

"""
from __future__ import print_function, division

import os
import pickle

import numpy as np
import astropy.units as u
from astropy import constants
from astropy.io import fits
from scipy.ndimage.filters import gaussian_filter1d
import matplotlib.pyplot as plt
from astropy.table import Table, hstack
from scipy.special import legendre
import pymc3 as pm

from spectres import spectres

import context
from run_ppxf import pPXF
from misc import array_from_header

def prepare_spectra(outw1, outw2, dw, dataset="MUSE-DEEP", redo=False,
                velscale=None, sigma=350):
    """ Prepare spectra for CSP modeling """
    velscale = context.velscale if velscale is None else velscale
    targetSN = 70
    w1 = 4500
    w2 = 10000
    regrid = np.arange(outw1, outw2, dw)
    for field in context.fields:
        wdir = os.path.join(context.data_dir, dataset, field)
        data_dir = os.path.join(wdir, "ppxf_vel{}_w{}_{}_sn{}".format(int(
            velscale), w1, w2, targetSN))
        pkls = sorted([_ for _ in os.listdir(data_dir) if _.endswith(".pkl")])
        outdir = os.path.join(wdir, "spec1d_resamp_w{}_{}_dw{}".format(outw1,
                              outw2, dw))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for j, pkl in enumerate(pkls):
            outfile = os.path.join(outdir, pkl.replace(".pkl", ".fits"))
            if os.path.exists(outfile) and not redo:
                continue
            # print("Working with file {} ({}/{})".format(pkl, j + 1, len(pkls)))
            ####################################################################
            # Reading data from pPXF fitting
            with open(os.path.join(data_dir, pkl)) as f:
                pp = pickle.load(f)
            ####################################################################
            # Subtracting emission lines
            wave = pp.table["wave"]
            flux = pp.table["flux"] - pp.table["emission"]
            # Convolve spectrum to given sigma
            losvd = pp.sol[0]
            z = losvd[0] * u.km / u.s / constants.c
            if losvd[1] > sigma:
                continue
            sigma_diff = np.sqrt(sigma ** 2 - losvd[1] ** 2) / pp.velscale
            flux = gaussian_filter1d(flux, sigma_diff, mode="constant",
                                     cval=0.0)
            ####################################################################
            # De-redshift and resample of the spectrum
            w0 = wave / (1 + z)
            # Resampling the spectra
            fregrid = spectres(regrid, w0, flux)
            table = Table([regrid, fregrid], names=["wave", "flux"])
            table.write(outfile, format="fits", overwrite=True)
            ####################################################################

def prepare_templates(outw1, outw2, dw, sigma=350, redo=False, velscale=None,
                      sample=None):
    """ Resample templates for full spectral fitting. """
    velscale = context.velscale if velscale is None else velscale
    sample = "salpeter_regular" if sample is None else sample
    w1 = 4500
    w2 = 10000
    tempfile = os.path.join(context.home, "templates",
        "emiles_muse_vel{}_w{}_{}_{}.fits".format(int(velscale), w1, w2,
                                                  sample))
    output = os.path.join(context.home, "templates",
             "emiles_sigma_{}_w{}_{}_dw{}_{}.fits".format(sigma, outw1, outw2,
                                                          dw, sample))
    if os.path.exists(output) and not redo:
        templates = fits.getdata(output, 0)
        wave = fits.getdata(output, 1)
        params = fits.getdata(output, 2)
        return wave, params, templates
    wave = np.exp(array_from_header(tempfile, axis=1,
                                                  extension=0))
    ssps = fits.getdata(tempfile, 0)
    params = Table.read(tempfile, hdu=2)
    newwave = np.arange(outw1, outw2, dw)
    templates = np.zeros((len(ssps), len(newwave)))
    norms = np.zeros(len(ssps))
    for i in np.arange(len(ssps)):
        sigma_pix = sigma / velscale
        flux = gaussian_filter1d(ssps[i], sigma_pix, mode="constant",
                                 cval=0.0)
        norm = np.median(flux)
        flux /= norm
        templates[i] = spectres(newwave, wave, flux)
        norms[i] = norm
    norms = Table([norms], names=["norm"])
    params = hstack([params, norms])
    hdu1 = fits.PrimaryHDU(templates)
    hdu2 = fits.ImageHDU(newwave)
    hdu3 = fits.BinTableHDU(params)
    hdulist = fits.HDUList([hdu1, hdu2, hdu3])
    hdulist.writeto(output, overwrite=True)
    return newwave, params, templates

def csp_modeling(wave, flux, templates, dbname, redo=False, adegree=10):
    """ Model a CSP with bayesian model. """
    if os.path.exists(dbname) and not redo:
        return
    x = np.linspace(-1, 1, len(wave))
    apoly = np.zeros((adegree, len(x)))
    for i in range(adegree):
        apoly[i] = legendre(i)(x)
    with pm.Model() as model:
        flux0 = pm.Normal("f0", mu=1, sd=5)
        w = pm.Dirichlet("w", np.ones(len(templates)))
        wpoly = pm.Normal("wpoly", mu=0, sd=1, shape=adegree)
        bestfit = pm.Deterministic("bestfit", flux0 * (pm.math.dot(w.T, \
                  templates) + pm.math.dot(wpoly.T, apoly)))
        sigma = pm.Exponential("sigma", lam=1)
        likelihood = pm.Normal('like', mu=bestfit, sd = sigma, observed=flux)
        # pm.Cauchy("like", alpha=bestfit, beta=sigma, observed=flux)
    with model:
        trace = pm.sample(1000, tune=1000)
    results = {'model': model, "trace": trace}
    with open(dbname, 'wb') as buff:
        pickle.dump(results, buff)
    return

def plot(obswave, flux, norm, dbname, outw1, outw2, dw, velscale=None):
    """ Plot results from model. """
    velscale = context.velscale if velscale is None else velscale
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
             "emiles_sigma_350_w{}_{}_dw{}_salpeter_regular.fits".format(outw1,
                                                                    outw2, dw))
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
    ax1.plot(obswave, norm * flux)
    ax1.plot(obswave, norm * bestfit)
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
    ij = np.array(np.meshgrid(idxs, idxs)).reshape(-1, 2)
    for k, (i, j) in enumerate(ij):
        plt.subplot(npars, npars, k+1)
        if i > j:
            continue
        elif i == j:
            values = np.unique(params[parnames[i]])
            w = np.zeros(len(values))
            for l,val in enumerate(values):
                idx = np.where(params[parnames[l]] == val)[0]
                print(values, w)
                if not len(idx):
                    continue
                w[l] = np.sum(weights[idx] / params["norm"][idx])
            plt.bar(values, w)
            plt.show()
        else:
            pass



    plt.show()

def run(dataset = "MUSE-DEEP"):
    # Parameters for the resampling
    outw1 = 4700
    outw2 = 9100
    dw = 5
    prepare_spectra(outw1, outw2, dw, redo=False)
    wave, params, templates = prepare_templates(outw1, outw2, dw, redo=False)
    templates = np.array(templates, dtype=np.float)
    for field in context.fields:
        wdir = os.path.join(context.data_dir, dataset, field,
                        "spec1d_resamp_w{}_{}_dw{}".format(outw1, outw2, dw))
        kintable = os.path.join(context.data_dir, dataset, "tables",
                                "ppxf_results_vel30_sn70_w4500_10000.fits")
        kindata = Table.read(kintable)
        outdir = wdir.replace("spec1d", "models")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        specs = sorted(os.listdir(wdir))
        for spec in specs:
            print(spec)
            idx = np.where(kindata["SPEC"] == spec.split(".")[0])[0]
            v = kindata["V"][idx]
            z = v / constants.c.to("km/s")
            obswave = wave * u.angstrom * (1 + z)
            dbname = os.path.join(outdir, spec.replace(".fits", ".db"))
            data = Table.read(os.path.join(wdir, spec))
            flux = data["flux"]
            norm = np.nanmedian(flux)
            flux /= norm
            csp_modeling(obswave, flux, templates, dbname, redo=False)
            plot(obswave, flux, norm, dbname, outw1, outw2, dw)


if __name__ == "__main__":
    run()
