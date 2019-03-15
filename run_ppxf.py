# -*- coding: utf-8 -*-
"""
Forked in Hydra IMF from Hydra/MUSE on Feb 19, 2018

@author: Carlos Eduardo Barbosa

Run pPXF in data
"""
import os
import yaml

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants
from astropy.table import Table, vstack, hstack

from ppxf.ppxf import ppxf
from ppxf import ppxf_util

import context
import misc
from der_snr import DER_SNR

def run_ppxf(specs, w1, w2, sample, templates_file, velscale=None, redo=False):
    """ Running pPXF. """
    velscale = context.velscale if velscale is None else velscale
    ssp_templates = fits.getdata(templates_file, extname="SSPS").T
    params = Table.read(templates_file, hdu=1)
    nssps = ssp_templates.shape[1]
    logwave_temp = Table.read(templates_file, hdu=2)["loglam"].data
    wave_temp = np.exp(logwave_temp)
    start0 = [context.V, 100., 0., 0.]
    bounds = [[[1800., 5800.], [3., 800.], [-0.3, 0.3], [-0.3, 0.3]],
               [[1800., 5800.], [3., 80.]]]
    outdir = os.path.join(os.getcwd(),
                          "ppxf_vel{}_w{}_{}_{}".format(int(velscale),
                                                          w1, w2, sample))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
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
        der_sn = misc.snr(flux)[2]
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
                  quiet=False, mdegree=15, bounds=bounds)
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
        pp.flux_norm = flux_norm
        pp.name = name
        pp.srn = float(np.nanmedian(pp.galaxy) / \
                      np.nanstd(pp.galaxy - pp.bestfit))
        pp.sn = der_sn
        # Saving the weights of the bestfit
        wtable = hstack([params[params.colnames[:-1]], weights])
        wtable.write(outtable, overwrite=True)
        # Saving results and plot
        save(pp, outdir)
        plt.savefig(os.path.join(outdir, "{}_ppxf.png".format(name)),
                    dpi=250)
        plt.clf()

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
                 "chi2", "nonzero_ssps", "nssps", "flux_norm", "sn", "srn"]
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

if __name__ == '__main__':
    targetSN = 150
    w1 = 4500
    w2 = 10000
    sample = "bsf"
    velscale = context.velscale
    dataset = "MUSE-DEEP"
    tempfile = os.path.join(context.home, "templates",
               "emiles_muse_vel{}_w{}_{}_{}.fits".format(int(velscale), w1, w2,
                                                    sample))
    fields = context.fields[:1]
    for field in fields:
        wdir = os.path.join(context.get_data_dir(dataset), field,
                            "spec1d_FWHM2.95_sn{}".format(targetSN))
        os.chdir(wdir)
        specs = sorted([_ for _ in os.listdir(".") if _.endswith(".fits")])
        run_ppxf(specs, w1, w2, sample, tempfile, redo=False)