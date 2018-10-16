# -*- coding: utf-8 -*-
""" 

Created on 16/10/18

Author : Carlos Eduardo Barbosa

Use MUSE sky cubes to extract mean spectra to be used in the stellar
population fitting.

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
import matplotlib.pyplot as plt

import context
from source_extraction import run_sextractor

def get_cube_names():
    """ Read headers to decide the field and type of data of fits files. """
    output = os.path.join(wdir, "info.txt")
    if os.path.exists(output):
        table = Table.read(output, format="ascii")
        return table
    filenames = [_ for _ in os.listdir(wdir) if _.endswith(".fits") and
                 _.startswith("ADP")]
    ids, dtypes, fields = [], [], []
    for fname in filenames:
        obj = fits.getval(fname, "OBJECT", 0)
        id = obj.split("3311_")[1].split()
        dtype = "img" if len(id) == 2 else "cube"
        ids.append(id[0])
        dtypes.append(dtype)
    table = Table([ids, dtypes, filenames], names=["id", "dtype", "filename"])
    ids, cubes, imgs = [], [], []
    for id in np.unique(table["id"]):
        idx = np.where(table["id"] == id)
        t = table[idx]
        idx_cube = np.where(t["dtype"]=="cube")[0][0]
        idx_img = np.where(t["dtype"]=="img")[0][0]
        ids.append(id)
        cubes.append(t["filename"][idx_cube])
        imgs.append(t["filename"][idx_img])
    table = Table([ids, cubes, imgs], names=["id", "cube", "img"])
    table.write(output, format="ascii", overwrite=True)
    return table

def extract_sky_spectra(table):
    """ Performs the extraction of the sky spectra. """
    for field in table:
        print(field)
        segfile = "{}_segmentation.fits".format(field["id"])
        output = "sky_{}.fits".format(field["id"])
        if not os.path.exists(segfile):
            run_sextractor(field["img"], segmentation_file=segfile)
        img = fits.getdata(field["img"])
        mask = np.where(np.isnan(img))
        segmentation = fits.getdata(segfile)
        segmentation[mask] = 1
        segmentation[segmentation>0] = 1
        idy, idx = np.where(segmentation==0)
        ncombine = len(idx)
        hdr = fits.getheader(field["cube"], 1)
        wave = ((np.arange(hdr['NAXIS3']) + 1 - hdr['CRPIX3']) * hdr['CD3_3'] \
                + hdr['CRVAL3'])
        data = fits.getdata(field["cube"], 1)
        errdata = np.sqrt(fits.getdata(field["cube"], 2))
        specs = data[:,idy,idx]
        errors = errdata[:,idy,idx]
        errs = np.sqrt(np.nansum(errors**2, axis=1)) / ncombine
        combined = np.nanmean(specs, axis=1)
        tab = Table([wave, combined, errs], names=["wave", "flux", "fluxerr"])
        tab.write(output, format="fits", overwrite=True)



if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "sky")
    os.chdir(wdir)
    table = get_cube_names()
    extract_sky_spectra(table)
