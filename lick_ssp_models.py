# -*- coding: utf-8 -*-
""" 

Created on 20/03/18

Author : Carlos Eduardo Barbosa

Modeling the stellar populations of NGC 3311 using Chiara's SSP models

"""
from __future__ import print_function, division

import os

import numpy as np
import astropy.units as u
from astropy.table import Table
import matplotlib.pyplot as plt
import pymc3

import context
from prepare_templates import EMiles_models
from basket.lick.lick import Lick

def prepare_ssp_models(redo=False):
    """ Prepare models for the modeling. """
    emiles = EMiles_models()
    pars = np.meshgrid(emiles.values.exponents, emiles.values.ZH,
                       emiles.values.age, emiles.values.alphaFe,
                       emiles.values.NaFe)
    pars = np.column_stack([_.flatten() for _ in pars])
    ############################################################################
    # Selecting the wavelenghts to be used in the shortened spectra
    bandsfile = os.path.join(os.path.split(os.path.abspath(__file__))[0],
                                "tables/spindex_CS.dat")
    bandsz0 = np.loadtxt(bandsfile, usecols=(3,6)) * u.AA
    filename = os.path.join(emiles.path, emiles.get_filename(*pars[0]))
    data = Table.read(filename, format="fits")
    wave = data["wave"].quantity
    di = int(1000 / context.velscale)
    idxs = []
    for w1, w2 in bandsz0:
        idx = np.argwhere(np.logical_and(wave > w1, wave < w2))
        if not len(idx):
            continue
        i = np.arange(np.maximum(0, idx.min()-di),
                      np.minimum(idx.max()+di, len(wave)))
        idxs = np.union1d(idxs, i)
    goodidx = idxs.astype(int)
    bandsz0 = np.loadtxt(bandsfile, usecols=(3, 4, 1, 2, 5, 6)) * u.AA
    units_bin = np.loadtxt(bandsfile, usecols=(7,))
    units = np.where(units_bin, u.Unit("mag"), u.Unit("angstrom"))
    names = np.loadtxt(bandsfile, usecols=(8,), dtype=str)
    # Crop the spectra around the Lick indices
    for i, p in enumerate(pars):
        print(i, p)
        filename = os.path.join(emiles.path, emiles.get_filename(*p))
        data = Table.read(filename, format="fits")
        wave = data["wave"].quantity
        ssp = data["flux"].quantity
        ll = Lick(wave, ssp, bandsz0)
        ll.classic_integration()
        raw_input()
        print(ll.Ia)

def model_ssp_single():
    """ Model SSP properties of single bins using one SSP model. """
    pass

if __name__ == "__main__":
    prepare_ssp_models(redo=False)
    model_ssp_single()