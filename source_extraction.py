# -*- coding: utf-8 -*-
"""

Created on 28/10/2017

@author: Carlos Eduardo Barbosa

Detection of sources in data and separation of bins prior to Voronoi
tesselation

"""
from __future__ import division, print_function
import os

import pyregion
import numpy as np
from astropy.io import fits
from astropy.convolution import Gaussian2DKernel, convolve
from astropy.table import Table

import sewpy

import context

def make_unsharp_mask(img, cube, redo=False, output=None):
    """ Produces unsharp mask of a given image. """
    if output is None:
        output = "unsharp_mask_{}.fits".format(cube)
    print(os.getcwd())
    print(os.listdir("."))
    print(output)
    input(os.path.exists(output))
    if os.path.exists(output) and not redo:
        return output
    kernel = Gaussian2DKernel(5)
    data = fits.getdata(img)
    smooth = convolve(data, kernel, normalize_kernel=False)
    unsharp_mask = data - smooth
    fits.writeto(output, unsharp_mask, overwrite=True)
    return output

def mask_regions(img, cube,redo=False):
    """ Mask regions marked in file mask.reg made in ds9. """
    filename = "mask_{}.reg".format(cube)
    outfile = "masked_{}.fits".format(cube)
    if not os.path.exists(filename):
        return img
    if os.path.exists(outfile) and not redo:
        return outfile
    r = pyregion.open(filename)
    data = fits.getdata(img)
    for i, region in enumerate(r.get_filter()):
        mask = region.mask(data.shape)
        data[mask] = np.nan
    hdu = fits.PrimaryHDU(data)
    hdu.writeto(outfile, overwrite=True)
    return outfile

def run_sextractor(img, cube, redo=False, outfile=None, segmentation_file=None,
                   save=False):
    """ Produces a catalogue of sources in a given field. """
    if outfile is None:
        outfile = "sexcat_{}.fits".format(cube)
    if segmentation_file is None:
        segmentation_file = "segmentation_{}.fits".format(cube)
    if os.path.exists(outfile) and not redo:
        return outfile
    params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
               "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO"]
    config = {"CHECKIMAGE_TYPE": "SEGMENTATION",
              "CHECKIMAGE_NAME": segmentation_file,
              "DETECT_THRESH" : 1.5, "DEBLEND_NTHRESH": 64}
    sew = sewpy.SEW(config=config, sexpath="sextractor", params=params)
    cat = sew(img)
    cat["table"].write(outfile, format="fits", overwrite=True)
    return outfile

def mask_background(img, cat, cube, redo=False, output=None):
    """ Produces segmentation image with bins for detected sources using
    elliptical regions. """
    if output is None:
        output = "mask_{}.fits".format(cube)
    if os.path.exists(output) and not redo:
        return output
    data = fits.getdata(img)
    ydim, xdim = data.shape
    xx, yy = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))
    table = Table.read(cat, 1)
    ignore = ignore_sources(cube)
    idx = np.array([i for i,x in enumerate(table["NUMBER"]) if x not in ignore])
    table = table[idx]
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
    d[segmentation==0] = np.nan
    hdu = fits.PrimaryHDU(d)
    hdu.writeto(output, overwrite=True)
    return output

def calc_isophotes(x, y, x0, y0, PA, q):
    """ Calculate isophotes for a given component. """
    x = np.copy(x) - x0
    y = np.copy(y) - y0
    shape = x.shape
    theta = np.radians(PA)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[s, c], [-c, s]])
    xy = np.dot(np.column_stack((x.flatten(), y.flatten())), rot).T
    x = np.reshape(xy[0], newshape=shape)
    y = np.reshape(xy[1], newshape=shape)
    return np.sqrt(np.power(x, 2) + np.power(y / q, 2))

def ignore_sources(cubes):
    if cubes == "cube2":
        return []
    else:
        return []

if __name__ == "__main__":
    cubes = context.obs
    data_dir = os.path.join(context.data_dir, "MUSE")
    os.chdir(data_dir)
    for cube in cubes:
        os.chdir(os.path.join(data_dir))
        imgname, cubename = context.get_field_files(cube)
        segmentation_file = "segmentation_{}.fits".format(cube)
        sexcat = run_sextractor(imgname, cube,
                                segmentation_file=segmentation_file, redo=False)
        segmentation = fits.getdata(segmentation_file).astype(np.float)
        segmentation[segmentation==0] = np.nan
        hdu = fits.PrimaryHDU(segmentation)
        hdu.writeto("mask_{}.fits".format(cube), overwrite=True)
