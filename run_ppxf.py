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
from spectres import spectres

import context
import misc
from der_snr import DER_SNR

def run_ppxf(specs, templates_file, outdir, velscale=None, redo=False, V0=None):
    """ Running pPXF. """
    velscale = context.velscale if velscale is None else velscale
    V0 = context.V if V0 is None else V0
    # Reading templates
    ssp_templates = fits.getdata(templates_file, extname="SSPS").T
    params = Table.read(templates_file, hdu=1)
    nssps = ssp_templates.shape[1]
    logwave_temp = Table.read(templates_file, hdu=2)["loglam"].data
    wave_temp = np.exp(logwave_temp)
    # Use first spectrum to set emission lines
    start0 = [V0, 100., 0., 0.]
    bounds0 = [[V0 - 2000., V0 + 2000], [velscale/10, 800.]]
    for spec in specs:
        print("Processing spectrum {}".format(spec))
        name = spec.replace(".fits", "")
        outyaml = os.path.join(outdir, "{}.yaml".format(name))
        if os.path.exists(outyaml) and not redo:
            continue
        table = Table.read(spec)
        wave_lin = table["wave"]
        flux = table["flux"]
        fluxerr = table["fluxerr"]
        # Removing red part of the spectrum
        idx = np.where(wave_lin < 7000)[0]
        wave_lin = wave_lin[idx]
        flux = flux[idx]
        fluxerr = fluxerr[idx]
        der_sn = misc.snr(flux)[2]
        data_sn = np.nanmedian(flux / fluxerr)
        ###################################################################
        # Rebinning the data to a logarithmic scale for ppxf
        wave_range = [wave_lin[0], wave_lin[-1]]
        logwave = ppxf_util.log_rebin(wave_range, flux, velscale=velscale)[1]
        wave = np.exp(logwave)
        wave = wave[(wave > wave_lin[0]) & (wave < wave_lin[-1])][1:-1]
        flux, fluxerr = spectres(wave, wave_lin, flux, spec_errs=fluxerr)
        ####################################################################
        # Setting up the gas templates
        gas_templates, line_names, line_wave = \
            ppxf_util.emission_lines(logwave_temp,
                                     [wave_lin[0], wave_lin[-1]], 2.95)
        ngas = gas_templates.shape[1]
        ####################################################################
        # Masking bad pixels
        skylines = np.array([4785, 5577, 5889, 6300, 6360, 6863])
        goodpixels = np.arange(len(wave))
        for line in skylines:
            sky = np.argwhere((wave < line - 10) | (wave > line + 10)).ravel()
            goodpixels = np.intersect1d(goodpixels, sky)
        # Making goodpixels mask
        goodpixels = np.intersect1d(goodpixels, np.where(np.isfinite(flux))[0])
        goodpixels = np.intersect1d(goodpixels, np.where(np.isfinite(
            fluxerr))[0])
        # Cleaning input spectrum
        fluxerr[~np.isfinite(fluxerr)] = np.nanmax(fluxerr)
        flux[~np.isfinite(flux)] = 0.
        ########################################################################
        # Preparing the fit
        dv = (logwave_temp[0] - logwave[0]) * \
             constants.c.to("km/s").value
        templates = np.column_stack((ssp_templates, gas_templates))
        components = np.hstack((np.zeros(nssps), np.arange(ngas)+1)).astype(
            np.int)
        gas_component = components > 0
        start = [start0[:2]] * (ngas + 1)
        bounds = [bounds0] * (ngas + 1)
        moments = [2] * (ngas + 1)
        ########################################################################
        # Fitting with two components
        pp = ppxf(templates, flux, fluxerr, velscale=velscale,
                  plot=True, moments=moments, start=start, vsyst=dv,
                  lam=wave, component=components, mdegree=-1,
                  gas_component=gas_component, gas_names=line_names,
                  quiet=False, degree=15, bounds=bounds, goodpixels=goodpixels)
        plt.savefig(os.path.join(outdir, "{}.png".format(name)), dpi=250)
        plt.close()
        pp.name = name
        # Saving results and plot
        save(pp, outdir)

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
                 "ncomp", "chi2"]
    # Chi2 is a astropy.unit.quantity object, we have to make it a scalar
    pp.chi2 = float(pp.chi2)
    for key in save_keys:
        ppdict[key] = getattr(pp, key)
    klist = ["V", "sigma"]
    for j, sol in enumerate(pp.sol):
        for i in range(len(sol)):
            ppdict["{}_{}".format(klist[i], j)] = float(sol[i])
            ppdict["{}err_{}".format(klist[i], j)] = float(pp.error[j][i])
    with open(os.path.join(outdir, "{}.yaml".format(pp.name)), "w") as f:
        yaml.dump(ppdict, f, default_flow_style=False)
    # Saving table with emission lines
    gas = pp.gas_component
    emtable = []
    for j, comp in enumerate(pp.component[gas]):
        t = Table()
        t["name"] = [ pp.gas_names[j]]
        t["flux"] = [pp.gas_flux[j]]
        t["fluxerr"] = [pp.gas_flux_error[j]]
        t["V"] = [pp.sol[comp][0]]
        t["Verr"] = [pp.error[comp][0]]
        t["sigma"] = [pp.sol[comp][1]]
        t["sigmaerr"] = [pp.error[comp][1]]
        emtable.append(t)
    emtable = vstack(emtable)
    emtable.write(os.path.join(outdir, "{}_emission_lines.fits".format(
                  pp.name)), overwrite=True)

def make_table(direc, output):
    """ Read all yaml files in a ppf directory to one make table for all
    bins. """
    filenames = sorted([_ for _ in os.listdir(direc) if _.endswith(".yaml")])
    keys = ["name", "V_0", "Verr_0", "sigma_0", "sigmaerr_0", "der_sn"]
    names = {"name": "spec", "V_0": "V", "Verr_0": "Verr",
             "sigma_0": "sigma", "sigmaerr_0": "sigmaerr", "der_sn": "SNR"}
    outtable = []
    for fname in filenames:
        with open(os.path.join(direc, fname)) as f:
            props = yaml.load(f)
        data = Table([[props[k]] for k in keys], names=[names[k] for k in keys])
        outtable.append(data)
    outtable = vstack(outtable)
    outtable.write(output, format="fits", overwrite=True)

if __name__ == '__main__':
    targetSN = 100
    sample = "kinematics"
    velscale = context.velscale
    tempfile = os.path.join(context.data_dir, "templates",
               "emiles_vel{}_{}_fwhm2.95.fits".format(int(velscale), sample))
    wdir = os.path.join(context.data_dir, "MUSE/sn{}/sci".format(targetSN))
    os.chdir(wdir)
    outdir = os.path.join(os.path.split(wdir)[0], "ppxf")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    specs = sorted([_ for _ in os.listdir(".") if _.endswith(".fits")])
    run_ppxf(specs, tempfile, outdir, redo=False)