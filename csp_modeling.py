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
            print("Working with file {} ({}/{})".format(pkl, j + 1, len(pkls)))
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
    return wave, params, templates

def csp_modeling(obs, templates, dbname, redo=False,):
    """ Model a CSP with bayesian model. """
    if os.path.exists(dbname) and not redo:
        return dbname
    with pm.Model() as model:
        w = pm.Dirichlet("w", np.ones(len(templates)))
        bestfit = pm.math.dot(w.T, templates)
        sigma = pm.Exponential("sigma", lam=1)
        likelihood = pm.Normal('like', mu=bestfit, sd = sigma, observed=obs)
    with model:
        trace = pm.sample(1000, tune=1000)
    results = {'model': model, "trace": trace}
    with open(dbname, 'wb') as buff:
        pickle.dump(results, buff)
    return

def run(dataset = "MUSE-DEEP"):
    # Parameters for the resampling
    outw1 = 4700
    outw2 = 9200
    dw = 10
    prepare_spectra(outw1, outw2, dw, redo=False)
    wave, params, templates = prepare_templates(outw1, outw2, dw, redo=True)
    for field in context.fields:
        wdir = os.path.join(context.data_dir, dataset, field,
                        "spec1d_resamp_w{}_{}_dw{}".format(outw1, outw2, dw))
        specs = sorted(os.listdir(wdir))
        for spec in specs:
            data = Table.read(os.path.join(wdir, spec))
            plt.plot(wave, data["flux"])
            plt.plot(wave, templates[-1] * np.median(data["flux"] /
                                                     np.median(templates[-1])))
            plt.show()


if __name__ == "__main__":
    run()
