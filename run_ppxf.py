# -*- coding: utf-8 -*-
"""
Forked in Hydra IMF from Hydra/MUSE on Feb 19, 2018

@author: Carlos Eduardo Barbosa

Run pPXF in data
"""
import os
import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import astropy.units as u
from astropy.stats import sigma_clip
from astropy.io import fits
from astropy import constants
from astropy.table import Table, vstack, hstack
from specutils.io.read_fits import read_fits_spectrum1d

from ppxf.ppxf import ppxf, reddening_curve
import ppxf.ppxf_util as util

import context
from misc import array_from_header, snr
from geomfov import get_geom


class pPXF():
    """ Class to read pPXF pkl files """
    def __init__(self, pp, velscale):
        self.__dict__ = pp.__dict__.copy()
        self.dw = 1.25 # Angstrom / pixel
        self.calc_arrays()
        self.calc_sn()
        return

    def calc_arrays(self):
        """ Calculate the different useful arrays."""
        # Slice matrix into components
        self.compmatrix = np.copy(self.matrix)
        self.m_poly = self.matrix[:,:self.degree + 1]
        self.matrix = self.matrix[:,self.degree + 1:]
        self.m_ssps = self.matrix[:,:self.ntemplates]
        self.matrix = self.matrix[:,self.ntemplates:]
        self.m_gas = self.matrix[:,:self.ngas]
        self.matrix = self.matrix[:,self.ngas:]
        self.m_sky = self.matrix
        # Slice weights
        if hasattr(self, "polyweights"):
            self.w_poly = self.polyweights
            self.poly = self.m_poly.dot(self.w_poly)
        else:
            self.poly = np.zeros_like(self.galaxy)
        if hasattr(self, "mpolyweights"):
            x = np.linspace(-1, 1, len(self.galaxy))
            self.mpoly = np.polynomial.legendre.legval(x, np.append(1,
                                                       self.mpolyweights))
        else:
            self.mpoly = np.ones_like(self.galaxy)
        if self.reddening is not None:
            self.extinction = reddening_curve(self.lam, self.reddening)
        else:
            self.extinction = np.ones_like(self.galaxy)
        self.w_ssps = self.weights[:self.ntemplates]
        self.weights = self.weights[self.ntemplates:]
        self.w_gas = self.weights[:self.ngas]
        self.weights = self.weights[self.ngas:]
        self.w_sky = self.weights
        # Calculating components
        self.ssps = self.m_ssps.dot(self.w_ssps)
        self.gas = self.m_gas.dot(self.w_gas)
        self.bestsky = self.m_sky.dot(self.w_sky)
        return

    def calc_sn(self):
        """ Calculates the S/N ratio of a spectra. """
        self.signal = np.nanmedian(self.galaxy[self.goodpixels])
        self.meannoise = np.nanstd(sigma_clip(self.galaxy[self.goodpixels] -
                                              self.bestfit[self.goodpixels],
                                              sigma=5))
        self.sn = self.signal / self.meannoise
        return

    def mc_errors(self, nsim=200):
        """ Calculate the errors using MC simulations"""
        errs = np.zeros((nsim, len(self.error)))
        for i in range(nsim):
            y = self.bestfit + np.random.normal(0, self.noise,
                                                len(self.galaxy))

            noise = np.ones_like(self.galaxy) * self.noise
            sim = ppxf(self.bestfit_unbroad, y, noise, velscale,
                       [0, self.sol[1]],
                       goodpixels=self.goodpixels, plot=False, moments=4,
                       degree=-1, mdegree=-1,
                       vsyst=self.vsyst, lam=self.lam, quiet=True, bias=0.)
            errs[i] = sim.sol
        median = np.ma.median(errs, axis=0)
        error = 1.4826 * np.ma.median(np.ma.abs(errs - median), axis=0)
        # Here I am using always the maximum error between the simulated
        # and the values given by pPXF.
        self.error = np.maximum(error, self.error)
        return

    def plot(self, output=None, fignumber=1):
        """ Plot pPXF run in a output file"""
        if self.ncomp > 1:
            sol = self.sol[0]
            error = self.error[0]
            sol2 = self.sol[1]
            error2 = self.error[1]
        else:
            sol = self.sol
            error = self.error
            sol2 = sol
            error2 = error
        plt.figure(fignumber, figsize=(5,4.2))
        plt.clf()
        gs = gridspec.GridSpec(2, 1, height_ratios=[2.5,1])
        ax = plt.subplot(gs[0])
        ax.minorticks_on()
        ax.plot(self.w[self.goodpixels], self.galaxy[self.goodpixels], "-",
                label="Data (S/N={0:.1f})".format(self.sn))
        ax.plot(self.w[self.goodpixels], self.bestfit[self.goodpixels], "-",
                label="SSPs: V={0:.0f} km/s, $\sigma$={1:.0f} km/s".format(
                    sol[0], sol[1]))
        ax.xaxis.set_ticklabels([])
        if self.ncomp == 2:
            ax.plot(self.w[self.goodpixels], self.gas[self.goodpixels], "-",
                    label="Emission: V={0:.0f} km/s, "
                          "$\sigma$={1:.0f} km/s".format(sol2[0],sol2[1]))
        ax.set_xlim(self.w[self.goodpixels][0], self.w[self.goodpixels][-1])
        ax.set_ylim(None, 1.1 * self.bestfit[self.goodpixels].max())
        leg = plt.legend(loc=3, prop={"size":10}, title="Field {0[0]}, "
                                                        "Bin {0[2]}".format(
            self.name[5:].split("_")),
                         frameon=False)
        # leg.get_frame().set_linewidth(0.0)
        plt.axhline(y=0, ls="--", c="k")
        plt.ylabel(r"Flux ($10^{-20}$ erg s$^{-1}$ cm$^{-2} \AA^{-1}$)",
                   size=12)
        ax1 = plt.subplot(gs[1])
        ax1.minorticks_on()
        ax1.set_xlim(self.w[0], self.w[-1])
        ax1.plot(self.w[self.goodpixels], (self.galaxy[self.goodpixels] - \
                 self.bestfit[self.goodpixels]), "-",
                 label="$\chi^2=${0:.2f}".format(self.chi2))
        ax1.plot(self.w[self.goodpixels], self.noise[self.goodpixels], "--k",
                 lw=0.5)
        ax1.plot(self.w[self.goodpixels], -self.noise[self.goodpixels], "--k",
                 lw=0.5)
        leg2 = plt.legend(loc=2, prop={"size":8})
        ax1.axhline(y=0, ls="--", c="k")
        # ax1.set_ylim(-8 * self.meannoise, 8 * self.meannoise)
        ax1.set_xlabel(r"$\lambda$ ($\AA$)", size=12)
        ax1.set_ylabel(r"$\Delta$Flux", size=12)
        ax1.set_xlim(self.w[self.goodpixels][0], self.w[self.goodpixels][-1])
        gs.update(hspace=0.075, left=0.11, bottom=0.11, top=0.98, right=0.98)
        if output is not None:
            plt.savefig(output, dpi=250)
        return

def ppsave(pp, outroot="logs/out"):
    """ Produces output files for a ppxf object. """
    arrays = ["matrix", "w", "bestfit", "goodpixels", "galaxy", "noise"]
    delattr(pp, "star_rfft")
    delattr(pp, "star")
    hdus = []
    for i,att in enumerate(arrays):
        if i == 0:
            hdus.append(fits.PrimaryHDU(getattr(pp, att)))
        else:
            hdus.append(fits.ImageHDU(getattr(pp, att)))
        delattr(pp, att)
    hdulist = fits.HDUList(hdus)
    hdulist.writeto(os.path.join(outroot, "{}.fits".format(pp.name)),
                    overwrite=True)
    with open(os.path.join(outroot, "{}.pkl".format(pp.name)) , "w") as f:
        pickle.dump(pp, f)
    return

def ppload(name, path):
    """ Read ppxf arrays. """
    with open(os.path.join(path, "{}.pkl".format(name))) as f:
        pp = pickle.load(f)
    arrays = ["matrix", "w", "bestfit", "goodpixels", "galaxy", "noise"]
    for i, item in enumerate(arrays):
        setattr(pp, item, fits.getdata(os.path.join(path, "{}.fits".format(
            pp.name)), i))
    return pp

def run_ppxf(fields, w1, w2, targetSN, tempfile,
             velscale=None, redo=False, ncomp=2, only_halo=False, bins=None,
             dataset=None, **kwargs):
    """ New function to run pPXF. """
    if velscale is None:
        velscale = context.velscale
    if dataset is None:
        dataset = "MUSE-DEEP"
    logwave_temp = array_from_header(tempfile, axis=1, extension=0)
    wave_temp = np.exp(logwave_temp)
    stars = fits.getdata(tempfile, 0)
    emission = fits.getdata(tempfile, 1)
    params = fits.getdata(tempfile, 2)
    ngas = len(emission)
    nstars = len(stars)
    ##########################################################################
    # Set components
    if ncomp == 1:
        templates = stars.T
        components = np.zeros(nstars, dtype=int)
        kwargs["component"] = components
    elif ncomp == 2:
        templates = np.column_stack((stars.T, emission.T))
        components = np.hstack((np.zeros(nstars), np.ones(ngas))).astype(int)
        kwargs["component"] = components
    ##########################################################################
    for field in fields:
        print "Working on Field {0}".format(field[-1])
        os.chdir(os.path.join(context.data_dir, dataset, field,
                              "spec1d_FWHM2.95"))
        logdir = os.path.join(context.data_dir, dataset, field,
                              "ppxf_vel{}_w{}_{}_sn{}".format(int(velscale),
                               w1, w2, targetSN))
        if not os.path.exists(logdir):
            os.mkdir(logdir)
        filenames = sorted(os.listdir("."))
        ######################################################################
        for i, fname in enumerate(filenames):
            name = fname.replace(".fits", "")
            if os.path.exists(os.path.join(logdir, fname)) and not redo:
                continue
            print("=" * 80)
            print("PPXF run {0}/{1}".format(i+1, len(filenames)))
            print("=" * 80)
            spec = read_fits_spectrum1d(fname)
            ###################################################################
            # Trim spectrum according to templates
            idx = np.argwhere(np.logical_and(
                              spec.wavelength.value > wave_temp[0],
                              spec.wavelength.value < wave_temp[-1]))
            wave = spec.wavelength[idx].T[0].value
            flux = spec.flux[idx].T[0].value
            ###################################################################
            signal, noise, sn = snr(flux)
            galaxy, logLam, vtemp = util.log_rebin([wave[0], wave[-1]],
                                                   flux, velscale=velscale)
            dv = (logwave_temp[0]-logLam[0]) * constants.c.to("km/s").value
            noise =  np.ones_like(galaxy) * noise
            kwargs["lam"] = wave
            ###################################################################
            # Masking bad pixels
            skylines = np.array([4785, 5577,5889, 6300, 6863])
            goodpixels = np.arange(len(wave))
            for line in skylines:
                sky = np.argwhere((wave < line - 15) | (wave > line +
                                                        15)).ravel()
                goodpixels = np.intersect1d(goodpixels, sky)
            kwargs["goodpixels"] = goodpixels
            ###################################################################
            kwargs["vsyst"] = dv
            # First fitting
            pp = ppxf(templates, galaxy, noise, velscale, **kwargs)
            title = "Field {0} Bin {1}".format(field[-1], bin)
            pp.name = name
            # Adding other things to the pp object
            pp.has_emission = True
            pp.dv = dv
            pp.w = np.exp(logLam)
            pp.velscale = velscale
            pp.ngas = ngas
            pp.ntemplates = nstars
            pp.templates = 0
            pp.name = name
            pp.title = title
            ppsave(pp, outroot=logdir)
    return

def make_table(fields, w1, w2, targetSN, dataset="MUSE-DEEP",
               velscale=None, redo=True):
    """ Make table with results. """
    if velscale is None:
        velscale = context.velscale
    output = os.path.join(context.data_dir, dataset, "tables",
                          "ppxf_results_vel{}_sn{}_w{}_{}.fits".format(int(
                              velscale), targetSN, w1, w2))
    if os.path.exists(output) and not redo:
        return
    names, sols, errors, chi2s, sns = [], [], [], [], []
    adegrees, mdegrees = [], []
    geoms = []
    for field in fields:
        print "Producing summary for Field {0}".format(field[-1])
        geoms.append(get_geom(field, targetSN))
        logdir = os.path.join(context.data_dir, dataset, field,
                              "ppxf_vel{}_w{}_{}_sn{}".format(int(velscale),
                               w1, w2, targetSN))
        os.chdir(logdir)
        fitsfiles = sorted([x for x in os.listdir(".") if x.endswith("fits")])
        for i, fname in enumerate(fitsfiles):
            print " Processing pPXF solution {0} / {1}".format(i+1,
                                                               len(fitsfiles))
            pp = ppload(fname.replace(".fits", ""), logdir)
            pp = pPXF(pp, velscale)
            sol = pp.sol if pp.ncomp == 1 else pp.sol[0]
            sol[0] += context.vhelio[field]
            sols.append(sol)
            error = pp.error if pp.ncomp == 1 else pp.error[0]
            errors.append(error)
            chi2s.append(pp.chi2)
            sns.append(pp.sn)
            adegrees.append(pp.degree)
            mdegrees.append(pp.mdegree)
            names.append(fname.replace(".fits", ""))
    geoms = vstack(geoms)
    sols = np.array(sols).T
    errors = np.array(errors).T
    kintable = Table(data=[names, sols[0] * u.km / u.s , errors[0] * u.km /
                           u.s, sols[1] * u.km / u.s, errors[1] * u.km / u.s,
                           sols[2], errors[2], sols[3], errors[3], chi2s,
                           sns, adegrees, mdegrees], \
                     names=["SPEC", "V", "Verr", "SIGMA", "SIGMAerr", "H3",
                            "H3err", "H4", "H4err", "CHI2", "S/N", "ADEGREE",
                            "MDEGREE"])
    results = hstack([kintable, geoms])
    results.write(output, format="fits", overwrite=True)
    return


def make_plots(fields, targetSN, w1, w2, redo=False, dataset=None,
               velscale=None):
    """ Make plot of all fits. """
    if dataset is None:
        dataset = "MUSE-DEEP"
    if velscale is None:
        velscale = context.velscale
    for field in fields:
        logdir = os.path.join(context.data_dir, dataset, field,
                              "ppxf_vel{}_w{}_{}_sn{}".format(int(velscale),
                               w1, w2, targetSN))
        os.chdir(logdir)
        fitsfiles = sorted([x for x in os.listdir(".") if x.endswith("fits")])
        for i, fname in enumerate(fitsfiles):
            print " Processing pPXF solution {0} / {1}".format(i+1,
                                                               len(fitsfiles))
            pp = ppload(fname.replace(".fits", ""), logdir)
            pp = pPXF(pp, velscale)
            pp.plot(output=fname.replace(".fits", ".png"))
            plt.clf()

def run_stellar_populations(fields, targetSN, w1, w2,
                            sampling=None, velscale=None, redo=False,
                            dataset=None):
    """ Run pPXF on binned data using stellar population templates"""
    if sampling is None:
        sampling = "salpeter_regular"
    if velscale is None:
        velscale = context.velscale
    if dataset is None:
        dataset = "MUSE-DEEP"
    tempfile = os.path.join(context.home, "templates",
               "emiles_muse_vel{}_w{}_{}_{}.fits".format(int(velscale), w1, w2,
                                                    sampling))
    bounds = np.array([[[1800., 5800.], [3., 800.], [-0.3, 0.3], [-0.3, 0.3]],
                       [[1800., 5800.], [3., 80.], [-0.3, 0.3], [-0.3, 0.3]]])
    kwargs = {"start" :  np.array([[3800, 50, 0., 0.], [3800, 50., 0, 0]]),
              "plot" : False, "moments" : [4, 4], "degree" : 12,
              "mdegree" : 0, "reddening" : None, "clean" : False,
              "bounds" : bounds}
    run_ppxf(fields, w1, w2, targetSN, tempfile, redo=redo, ncomp=2, **kwargs)
    make_table(fields, w1, w2, targetSN, redo=redo)
    make_plots(fields, targetSN, w1, w2, redo=redo)
    return
if __name__ == '__main__':
    targetSN = 70
    w1 = 4500
    w2 = 5900
    ##########################################################################
    # Running stellar populations
    run_stellar_populations(context.fields, targetSN, w1, w2, redo=False)