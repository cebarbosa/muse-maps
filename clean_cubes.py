# -*- coding: utf-8 -*-
""" 

Created on 18/12/17

Author : Carlos Eduardo Barbosa

Use of ZAP (and possibly other routines) to clean the spectra from MUSE data
cubes in the Hydra IMF study in the MUSE dataset, where sky-offset observations
are available

"""
from __future__ import print_function, division
import os

import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack
from multiprocessing import cpu_count
import zap
import matplotlib.pyplot as plt

import context
from source_extraction import run_sextractor
from geomfov import calc_isophotes


def fits_info(direc):
    """ Print object information according to headers for all fits files
    inside a directory. """
    fitsfiles = sorted([_ for _ in os.listdir(direc) if _.endswith(".fits")])
    info = {}
    for fname in fitsfiles:
        h = fits.getheader(os.path.join(direc, fname))
        if "OBJECT" not in h.keys():
            continue
        obj = h["object"]
        info[obj] = fname
    cubes = dict([(x,y) for (x,y) in info.iteritems() if "white" not in x
                  and "SKY" not in x])
    table = []
    for cube in cubes:
        cubename = cubes[cube]
        skycube = info["SKY_for_{}".format(cube)]
        imgkey = [_ for _ in info if _.startswith(cube) and
                  _.endswith("(white)")][0]
        skyimgkey = "SKY_for_{}".format(imgkey)
        imgname = info[imgkey]
        skyimgname = info[skyimgkey]
        t = Table(data = [[cube,], [cubename,], [imgname,], [skycube,],
                           [skyimgname,]],
                  names=["name", "cube", "image", "sky", "skyimage"])
        table.append(t)
    table = vstack(table)
    return table

def mask_sources(img, cat, redo=False, output=None):
    """ Produces segmentation image with bins for detected sources using
    elliptical regions. """
    if output is None:
        output = "mask.fits"
    if os.path.exists(output) and not redo:
        return output
    data = fits.getdata(img)
    ydim, xdim = data.shape
    xx, yy = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))
    table = Table.read(cat, 1)
    axratio = table["B_IMAGE"] / table["A_IMAGE"]
    table = table[axratio > 0.4]
    segmentation = np.zeros_like(data)
    for source in table:
        R = calc_isophotes(xx, yy, source["X_IMAGE"], source["Y_IMAGE"], \
                           source["THETA_IMAGE"] - 90, source["B_IMAGE"] /
                           source["A_IMAGE"])
        Rmax = source["A_IMAGE"] * source["KRON_RADIUS"]
        segmentation += np.where(R <= Rmax, source["NUMBER"], 0.)
    d = np.copy(data)
    d[segmentation!=0] = np.nan
    hdu = fits.PrimaryHDU(d)
    hdu.writeto(output, overwrite=True)
    return output

def make_zap_mask(dataset, redo=False):
    """ Produces a mask to be used by ZAP based on field D. """
    output = "{}_zap_mask.fits".format(dataset["name"])
    if os.path.exists(output) and not redo:
        return output
    img = dataset["skyimage"]
    skycat = run_sextractor(img, redo=True,
             outfile="{}_skysources.fits".format(dataset["name"]))
    skymask = mask_sources(img, skycat, output="{}_skymask.fits".format(
              dataset["name"]))
    data = fits.getdata(skymask)
    idxsky = np.where(np.isfinite(data))
    idxmask = np.where(np.isnan(data))
    data[idxsky] = 0
    data[idxmask] = 1
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(output, overwrite=True)
    return output


if __name__ == "__main__":
    redo=False
    for field in context.fields:
        wdir = os.path.join(context.data_dir, "MUSE", field)
        os.chdir(wdir)
        datasets = fits_info(wdir)
        for dataset in datasets:
            outcube = "{}.fits".format(dataset["name"])
            data = fits.getdata(dataset["cube"])
            newdata = fits.getdata(outcube)
            plt.plot(data[:,235,212] - newdata[:,235,212], "-")
            plt.show()
            break


            # if os.path.exists(outcube) and not redo:
            #     continue
            # zapmask = make_zap_mask(dataset)
            # ncpu = cpu_count() -1
            # extSVD = zap.SVDoutput(dataset["sky"], mask=zapmask, ncpu=ncpu)
            # zap.process(dataset["cube"], outcubefits=outcube, extSVD=extSVD,
            #             ncpu=ncpu)
        break