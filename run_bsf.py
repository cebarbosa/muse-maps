# -*- coding: utf-8 -*-
"""

Created on 18/07/19

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import platform

import numpy as np
import astropy.units as u
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt
from numpy.polynomial import Legendre
import theano
import pymc3 as pm
import theano.tensor as tt
import extinction

from ppxf.ppxf_util import emission_lines

import context
import bsf

class SpecModel():
    def __init__(self, wave, velscale=None, test=False, nssps=1,
                 use_emission=False, porder=5):
        self.velscale = 50 * u.km / u.s if velscale is None else velscale
        self.wave = wave
        self.porder = porder
        # Templates have been already convolved to match the resolution of the
        # observations
        tempfile_extension = "bsf" if test is False else "test"
        templates_file = os.path.join(context.home, "templates",
                                 "emiles_muse_vel{}_w4500_10000_{}.fits".format(
                                  int(self.velscale.value), tempfile_extension))
        templates = fits.getdata(templates_file, ext=0)
        table = Table.read(templates_file, hdu=1)
        logwave = Table.read(templates_file, hdu=2)["loglam"].data
        twave = np.exp(logwave) * u.angstrom
        if use_emission is True:
            # TODO: work on cases with emission lines
            emission, line_names, line_wave = emission_lines(logwave,
                                                             [wave.value.min(),
                                                              wave.value.max()],
                                                             2.95)
            gas_templates = emission.T
            # gas_templates /= np.max(gas_templates, axis=1)[:,np.newaxis]
            em_components = np.ones(7)
        else:
            self.spec = bsf.SEDModel(twave, table, templates, nssps=nssps,
                               wave_out=self.wave, velscale=self.velscale)
            self.parnames = [_.split("_")[0] for _ in self.spec.parnames]
        # Making polynomial to slightly change continuum
        N = len(wave)
        self.x = np.linspace(-1, 1, N)
        self.poly = np.ones((porder, N))
        for i in range(porder):
            self.poly[i] = Legendre.basis(i+1)(self.x)
            self.parnames.append("a{}".format(i+1))
        self.grad_dim = (len(self.parnames), len(self.wave))
        self.nparams = self.spec.nparams + porder
        self.ssp_parameters = table.colnames

    def __call__(self, theta):
        p0, p1 = np.split(theta, [len(theta)- self.porder])
        return self.spec(p0) * (1. + np.dot(p1, self.poly))

    def gradient(self, theta):
        grad = np.zeros(self.grad_dim)
        p0, p1 = np.split(theta, [len(theta)- self.porder])
        spec = self.spec(p0)
        specgrad = self.spec.gradient(p0)
        poly = (1. + np.dot(p1, self.poly))
        grad[:len(theta)- self.porder] = specgrad * poly
        grad[len(theta)- self.porder:] = spec * self.poly
        return grad


def run_bsf(spec, test=False, nssps=1, redo=False, velscale=None,
            use_emission=False, outdir=None, draws=500):
    """ Routine to run BSF in a single spectrum"""
    name = spec.split(".")[0]
    outdir = os.path.join(os.getcwd(), "bsf") if outdir is None else outdir
    outdb = os.path.join(outdir, name)
    if os.path.exists(outdb) and not redo:
        return
    summary = "{}.csv".format(outdb)
    ############################################################################
    # Reading input data
    # Data from central region of NGC 3311 observed with MUSE
    # Resolution have been homogenized to 2.95 Angstrom
    data = Table.read(spec)
    wave = data["wave"].data.byteswap().newbyteorder()  * u.angstrom
    flam = data["flux"].data
    flamerr = data["fluxerr"].data
    plt.plot(wave, flam)
    ############################################################################
    # Make extinction correction
    CCM89 = extinction.ccm89(wave, context.Av, context.Rv)
    flam = extinction.remove(CCM89, flam)
    flamerr = extinction.remove(CCM89, flamerr)
    # Remove problematic bins
    # idx = np.where(np.isfinite(flux))[0]
    # flux = flux[idx]
    # fluxerr = fluxerr[idx]
    # wave = wave[idx]
    ############################################################################
    # Building parametric model for fitting
    porder = 10
    sed = SpecModel(wave, test=test, porder=porder)
    p0 = np.hstack([[0.1, 4., 1., 0., 5., 3500, 100], np.ones(porder) * 0.01])
    # Estimating flux
    m0 = -2.5 * np.log10(np.median(flam) / np.median(sed.spec.templates))
    # Making fitting
    model = pm.Model()
    with model:
        Av = pm.Exponential("Av", lam=1 / 0.2, testval=0.1)
        BNormal = pm.Bound(pm.Normal, lower=0)
        Rv = BNormal("Rv", mu=3.1, sd=1., testval=3.1)
        mag = pm.Normal("mag", mu=m0, sd=3., testval=m0)
        flux = pm.Deterministic("flux",
                                pm.math.exp(-0.4 * mag * np.log(10)))
        theta = [Av, Rv, flux]
        # Setting limits given by stellar populations
        ########################################################################
        for param in sed.ssp_parameters:
            vmin = sed.spec.params[param].min()
            vmax = sed.spec.params[param].max()
            vmean = 0.5 * (vmin + vmax)
            p = pm.Uniform(param, lower=vmin, upper=vmax, testval=vmean)
            theta.append(p)
        V = pm.Normal("V", mu=3800., sd=100., testval=3800.)
        theta.append(V)
        BoundHalfNormal = pm.Bound(pm.HalfNormal, lower=25)
        sigma = BoundHalfNormal("sigma", sd=np.sqrt(2) * 200)
        theta.append(sigma)
        for i in range(porder):
            a = pm.Normal("a{}".format(i+1), mu=0, sd=0.01)
            theta.append(a)
        nu = pm.Uniform("nu", lower=2.01, upper=50, testval=10.)
        theta.append(nu)
        theta = tt.as_tensor_variable(theta).T
        logl = bsf.LogLike(flam, wave, flamerr, sed, loglike="studt")
        # use a DensityDist
        pm.DensityDist('loglike', lambda v: logl(v),
                       observed={'v': theta})
        trace = pm.sample()
        df = pm.stats.summary(trace, alpha=0.3173)
        df.to_csv(summary)
        pm.save_trace(trace, outdb, overwrite=True)
    return

def main(targetSN=80, test=False):
    wdir = os.path.join(context.get_data_dir("MUSE-DEEP"), "fieldA",
                        "spec1d_FWHM2.95_sn{}".format(targetSN))
    os.chdir(wdir)
    s = "_test" if test is True else ""
    outdir = os.path.join(context.get_data_dir("MUSE-DEEP"), "fieldA",
                          "bsf_sn{}{}".format(targetSN, s))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    filenames = sorted([_ for _ in os.listdir(wdir) if _.endswith(".fits")])
    if platform.node() == "kadu-Inspiron-5557":
        for fname in filenames:
            run_bsf(fname, test=test, outdir=outdir)

if __name__ == "__main__":
    main()
