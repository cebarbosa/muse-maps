# -*- coding: utf-8 -*-
""" 

Created on 15/03/18

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import pickle

import numpy as np
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt

import context
from basket.lick.lick import Lick

import context
from run_ppxf import pPXF

def run_lick(w1, w2, targetSN, dataset="MUSE-DEEP", redo=False, velscale=None,
             nsim=200):
    """ Calculates Lick indices and uncertainties based on pPXF fitting. """
    velscale = context.velscale if velscale is None else velscale
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bandsz0 = np.loadtxt(bandsfile, usecols=(3,4,1,2,5,6)) * u.AA
    units_bin = np.loadtxt(bandsfile, usecols=(7,))
    units = np.where(units_bin, u.Unit("mag"), u.Unit("angstrom"))
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str)
    for field in context.fields:
        wdir = os.path.join(context.data_dir, dataset, field)
        data_dir = os.path.join(wdir, "ppxf_vel{}_w{}_{}_sn{}".format(int(
            velscale), w1, w2, targetSN))
        pkls = sorted([_ for _ in os.listdir(data_dir) if _.endswith(".pkl")])
        outdir = os.path.join(wdir, "lick")
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        for j, pkl in enumerate(pkls):
            print("Working with file {} ({}/{})".format(pkl, j+1, len(pkls)))
            output = os.path.join(outdir, pkl.replace(".pkl", "_nsim{"
                                                              "}.fits".format(nsim)))
            if os.path.exists(output) and not redo:
                continue
            with open(os.path.join(data_dir, pkl)) as f:
                pp = pickle.load(f)
            wave = pp.table["wave"] * u.AA
            flux = pp.table["flux"] - pp.table["emission"]
            noise = pp.table["flux"] - pp.table["bestfit"]
            losvd = pp.sol[0]
            losvderr = pp.error[0]
            lick = Lick(wave, flux, bandsz0, vel=losvd[0] * u.km / u.s,
                        units=units)
            lick.classic_integration()
            L = lick.classic
            Ia = lick.Ia
            Im = lick.Im
            R = lick.R
            veldist = np.random.normal(losvd[0], losvderr[0], nsim)
            Lsim= np.zeros((nsim, len(bandsz0)))
            Iasim = np.zeros((nsim, len(bandsz0)))
            Imsim = np.zeros((nsim, len(bandsz0)))
            Rsim = np.zeros((nsim, len(bandsz0)))
            for i, vel in enumerate(veldist):
                lsim = Lick(wave, flux + np.random.choice(noise), bandsz0,
                            vel=vel * u.km / u.s)
                lsim.classic_integration()
                Lsim[i] = lsim.classic
                Iasim[i] = lsim.Ia
                Imsim[i] = lsim.Im
                Rsim[i] = lsim.R
            Lerr = np.std(Lsim, axis=0)
            Iaerr = np.std(Iasim, axis=0)
            Imerr = np.std(Imsim, axis=0)
            Rerr = np.std(Rsim, axis=0)
            table = Table([names, L, Lerr, Ia, Iaerr, Im, Imerr, R, Rerr],
                          names=["name", "lick", "lickerr", "Ia",
                                 "Iaerr", "Im", "Imerr", "R", "Rerr"])
            table.write(output, format="fits", overwrite=True)

if __name__ == "__main__":
    targetSN = 70
    w1 = 4500
    w2 = 10000
    run_lick(w1, w2, targetSN, nsim=200)
