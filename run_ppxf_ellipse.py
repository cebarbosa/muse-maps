# -*- coding: utf-8 -*-
""" 

Created on 16/11/18

Author : Carlos Eduardo Barbosa

Run pPXF in spectra of galaxy NGC 3311 in ellipsse binning.

"""
from __future__ import print_function, division

import os
import yaml

import numpy as np
from astropy.io import fits
from astropy import constants
from astropy.stats import sigma_clip
from astropy.table import Table, hstack
import matplotlib.pyplot as plt
from ppxf import ppxf_util
from ppxf.ppxf import ppxf

import context
from misc import snr

def run_ppxf(velscale=None, w1=None, w2=None, sample=None, regul_err=None,
             library=None, version=None, redo=False):
    """ Run pPXF in all spectra. """
    velscale = context.velscale if velscale is None else velscale
    w1 = context.w1 if w1 is None else w1
    w2 = context.w2  if w2 is None else w2
    library = "miles" if library is None else library
    regul_err = 0.005 if regul_err is None else regul_err
    version = 0 if version is None else version
    if regul_err == 0:
        regul = 0
    else:
        regul = 1. / regul_err
    sample = "all" if sample is None else sample
    templates_file = os.path.join(context.home, "templates",
               "emiles_muse_vel{}_w{}_{}_{}.fits".format(int(velscale), w1, w2,
                                                    sample))
    ssp_templates = fits.getdata(templates_file, extname="SSPS").T
    params = Table.read(templates_file, hdu=1)
    # ssp_templates, reg_dim, params = make_regul_array(ssp_templates, params)
    nssps = ssp_templates.shape[1]
    logwave_temp = Table.read(templates_file, hdu=2)["loglam"].data
    wave_temp = np.exp(logwave_temp)
    start0 = [context.V, 100., 0., 0.]
    bounds = [[[1800., 5800.], [3., 800.], [-0.3, 0.3], [-0.3, 0.3]],
               [[1800., 5800.], [3., 80.]]]
    for field in context.fields:
        print(field)
        wdir = os.path.join(context.data_dir, "MUSE/combined", field,
                            "spec1d_ellipv{}_fwhm2.95".format(version))
        if not os.path.exists(wdir):
            continue
        os.chdir(wdir)
        # Creating output directory
        outdir = os.path.join(context.data_dir, "MUSE/combined", field,
                              "ppxf_ellipv{}_vel{}_{}_{}".format(version,
                                  velscale, library, sample))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        # Checking for the correct spectrum to be used.
        specs = sorted([_ for _ in os.listdir(".")])
        for spec in specs:
            # Reading the data in the files
            name = spec.replace(".fits", "")
            outtable = os.path.join(outdir, "{}_weights.fits".format(name))
            if os.path.exists(outtable) and not redo:
                continue
            print("Processing spectrum {}".format(name))
            table = Table.read(spec)
            wave = table["wave"]
            flux = table["flux"]
            fluxerr = table["fluxerr"]
            ###################################################################
            # Trim spectra to conform to the wavelenght range of the templates
            idx = np.argwhere(np.logical_and(
                              wave > wave_temp[0],
                              wave < wave_temp[-1]))
            wave = wave[idx].T[0]
            flux = flux[idx].T[0]
            fluxerr = fluxerr[idx].T[0]
            ####################################################################
            # Normalize spectrum to set regularization to a reasonable scale
            flux_norm = float(np.median(flux))
            flux /= flux_norm
            fluxerr /= flux_norm
            ####################################################################
            # Rebinning the data to a logarithmic scale for ppxf
            wave_range = [wave[0], wave[-1]]
            galaxy, logLam, vtemp = ppxf_util.log_rebin(wave_range,
                                                   flux, velscale=velscale)
            noise = ppxf_util.log_rebin(wave_range, fluxerr,
                                   velscale=velscale)[0]
            ####################################################################
            # Setting up the gas templates
            gas_templates, line_names, line_wave = \
                ppxf_util.emission_lines(logwave_temp,
                                         [wave[0], wave[-1]], 2.95)
            ngas = gas_templates.shape[1]
            ####################################################################
            # Preparing the fit
            start = [start0, start0[:2]]
            dv = (logwave_temp[0] - logLam[0]) * \
                 constants.c.to("km/s").value
            templates = np.column_stack((ssp_templates, gas_templates))
            components = np.hstack((np.zeros(nssps), np.ones(ngas))
                                   ).astype(np.int)
            gas_component = components > 0
            ########################################################################
            # Fitting with two components
            pp = ppxf(templates, galaxy, noise, velscale=velscale,
                      plot=True, moments=[4,2], start=start, vsyst=dv,
                      lam=np.exp(logLam), component=components, degree=-1,
                      gas_component=gas_component, gas_names=line_names,
                      quiet=False,  mdegree=40, bounds=bounds)
            # Calculating average stellar populations
            weights = Table([pp.weights[:nssps] * params["norm"]],
                            names=["mass_weight"])
            for colname in params.colnames[:-1]:
                mean = np.average(params[colname], weights=weights[
                    "mass_weight"].data)
                setattr(pp, colname, float(mean))
            # Including additional info in the pp object
            pp.nssps = nssps
            pp.nonzero_ssps = np.count_nonzero(weights)
            pp.regul_err = regul_err
            pp.flux_norm = flux_norm
            pp.name = name
            pp.sn = float(snr(pp.galaxy))
            # Saving the weights of the bestfit
            wtable = hstack([params[params.colnames[:-1]], weights])
            wtable.write(outtable, overwrite=True)
            # Saving results and plot
            save(pp, outdir)
            plt.savefig(os.path.join(outdir, "{}_ppxf.png".format(name)),
                        dpi=250)
            plt.clf()
            ####################################################################

def make_regul_array(ssp_templates, params):
    """ Make a regular regrid of templates to allow use of ppxf's regul"""
    ages = np.unique(params["age"].data)
    metals = np.unique(params["[Z/H]"].data)
    alphas = np.unique(params["[alpha/Fe]"].data)
    npixels = len(ssp_templates[0])
    regul_array = np.zeros((npixels, len(ages), len(metals), len(alphas)))
    newparams = []
    for i, age in enumerate(ages):
        for j, metal in enumerate(metals):
            for k, alpha in enumerate(alphas):
                idx = np.where((age==params["age"]) & (metal==params["[Z/H]"])\
                               & (alpha==params["[alpha/Fe]"]))[0][0]
                regul_array[:,i,j,k] = ssp_templates[idx]
                newparams.append([age, metal, alpha, params["norm"][idx]])
    newparams = np.array(newparams).T
    newparams = Table([newparams[0], newparams[1], newparams[2], newparams[3]],
                      names=["age", "[Z/H]", "[alpha/Fe]", "norm"])
    reg_dim = regul_array.shape[1:]
    regul_array = regul_array.reshape(regul_array.shape[0], -1)
    return regul_array, reg_dim, newparams

def save(pp, outdir):
    """ Save results from pPXF into files excluding fitting arrays. """
    array_keys = ["lam", "galaxy", "noise", "bestfit", "gas_bestfit",
                  "mpoly", "apoly"]
    array_keys = [_ for _ in array_keys if isinstance(getattr(pp, _),
                                                      np.ndarray)]
    table = Table([getattr(pp, key) for key in array_keys], names=array_keys)
    table.write(os.path.join(outdir, "{}_bestfit.fits".format(pp.name)),
                overwrite=True)
    ppdict = {}
    save_keys = ["name", "regul", "degree", "mdegree", "reddening", "clean",
                 "ncomp", "age", "[Z/H]", "[alpha/Fe]", "alpha", "[Na/Fe]",
                 "chi2", "nonzero_ssps", "nssps", "flux_norm", "regul_err",
                 "sn"]
    # Chi2 is a astropy.unit.quantity object, we have to make it a scalar
    pp.chi2 = float(pp.chi2)
    for key in save_keys:
        ppdict[key] = getattr(pp, key)
    klist = ["V", "sigma", "h3", "h4", "h5", "h6"]
    for j, sol in enumerate(pp.sol):
        for i in range(len(sol)):
            ppdict["{}_{}".format(klist[i], j)] = float(sol[i])
            ppdict["{}err_{}".format(klist[i], j)] = float(pp.error[j][i])

    with open(os.path.join(outdir, "{}.yaml".format(pp.name)), "w") as f:
        yaml.dump(ppdict, f, default_flow_style=False)

if __name__ == "__main__":
    run_ppxf(sample="bsf", redo=True)