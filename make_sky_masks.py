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

def make_sky_masks(table, redo=False):
    """ Select regions to be used as masks for ZAP. """
    for field in table:
        output = "skymask_{}.fits".format(field["id"])
        if os.path.exists(output) and not redo:
            continue
        segfile = "{}_segmentation.fits".format(field["id"])
        if not os.path.exists(segfile):
            run_sextractor(field["img"], segmentation_file=segfile)
        img = fits.getdata(field["img"])
        nans = np.where(np.isnan(img))
        mask = fits.getdata(segfile)
        mask[nans] = 1
        mask[mask>0] = 1
        hdu = fits.PrimaryHDU(mask)
        hdu.writeto(output, overwrite=True)

if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "sky")
    os.chdir(wdir)
    table = get_cube_names()
    make_sky_masks(table)
