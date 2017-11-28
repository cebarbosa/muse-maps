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
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import sewpy

import context

def make_unsharp_mask(img, redo=False):
    """ Produces unsharp mask of a given image. """
    output = "unsharp_mask.fits"
    if os.path.exists(output) and not redo:
        return output
    kernel = Gaussian2DKernel(5)
    data = fits.getdata(img)
    smooth = convolve(data, kernel, normalize_kernel=False)
    unsharp_mask = data - smooth
    fits.writeto(output, unsharp_mask, overwrite=True)
    return output

def mask_regions(img):
    """ Mask regions marked in file mask.reg made in ds9. """
    filename = "mask.reg"
    if not os.path.exists(filename):
        return img
    r = pyregion.open(filename)
    data = fits.getdata(img)
    for i, region in enumerate(r.get_filter()):
        mask = region.mask(data.shape)
        data[mask] = np.nan
    hdu = fits.PrimaryHDU(data)
    hdu.writeto("masked.fits", overwrite=True)
    return "masked.fits"

def run_sextractor(img, redo=False):
    """ Produces a catalogue of sources in a given field. """
    outfile = "sexcat.fits"
    if os.path.exists(outfile) and not redo:
        return outfile
    params = ["NUMBER", "X_IMAGE", "Y_IMAGE", "KRON_RADIUS", "ELLIPTICITY",
               "THETA_IMAGE", "A_IMAGE", "B_IMAGE", "MAG_AUTO"]
    config = {"CHECKIMAGE_TYPE": "SEGMENTATION",
                            "CHECKIMAGE_NAME": "segmentation.fits",
                            "DETECT_THRESH" : 1.5}
    sew = sewpy.SEW(config=config, sexpath="sex", params=params)
    cat = sew(img)
    cat["table"].write("sexcat.fits", format="fits", overwrite=True)
    return outfile

def mask_sources(img, cat, field):
    """ Produces segmentation image with bins for detected sources using
    elliptical regions. """
    data = fits.getdata(img)
    ydim, xdim = data.shape
    extent = np.array([1, xdim, 1, ydim])
    xx, yy = np.meshgrid(np.arange(1, xdim + 1), np.arange(1, ydim + 1))
    table = Table.read(cat, 1)
    ignore = ignore_sources(field)
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
    d[segmentation!=0] = np.nan
    hdu = fits.PrimaryHDU(d)
    hdu.writeto("masked_sources.fits", overwrite=True)
    vmax = np.percentile(d[np.isfinite(d)], 99.5)
    plt.subplot(131)
    plt.imshow(data, origin="bottom", vmax=vmax, vmin=0)
    plt.subplot(132)
    plt.imshow(segmentation, origin="bottom", vmax=table["NUMBER"][-1])
    plt.subplot(133)
    plt.imshow(d, origin="bottom", vmax=vmax, vmin=0)
    plt.show()

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

def ignore_sources(field):
    if field == "fieldA":
        return [113, 114]
    else:
        return []

def remove_sources(field):
    """ Remove bad sources. """
    if field == "fieldA":
        sources = [45,32,26,12,19,6,50,5,104,54,100,48,52,18,63,112,103,114,8,
                   49,46,42,38,31,13,7,4,30,44,16,17,1]
    elif field == "fieldB":
        sources = [9,45,2,3,1,24]
    elif field == "fieldC":
        sources = [23,3,4,5,12,90,75,77,85,71,82,24,21,11,44,50]
    elif field == "fieldD":
        sources = [73,78,8,64,66,68,24,23]
    img = pf.getdata("segmentation.fits")
    for source in sources:
        img[img==source] = 0
    pf.writeto("sources.fits", img, clobber=True)
    return

def include_regions():
    region_name = "newsources.reg"
    if not os.path.exists(region_name):
        return
    r = pyregion.open(region_name)
    img = pf.getdata("sources.fits")
    segments = pf.getdata("segmentation.fits")
    segs = segments.max()
    for i, region in enumerate(r.get_filter()):
        mask = region.mask(img.shape)
        img[mask] = segs + 1 + i
    pf.writeto("sources.fits", img, clobber=True)
    return

def modify_binning(field):
    if field in ["fieldA", "fieldB"]:
        return
    seg = pf.getdata("sources.fits")
    white = [x for x in os.listdir(".") if "(white)" in x][0]
    ra = wavelength_array(white, axis=1)
    dec = wavelength_array(white, axis=2)
    # Ofset to the center of NGC 3311
    ra -= ra0
    dec -= dec0
    # Convert to radians
    X = D * 1000 * np.deg2rad(ra)
    Y = D * 1000 * np.deg2rad(dec)
    xx, yy = np.meshgrid(X,Y)
    R = np.sqrt(xx**2 + yy**2)
    base = 10
    Rbins = 10 + 35 * np.logspace(0.3,1,4, base=base) / base
    Rbins = np.hstack((10, Rbins))
    for i,rbin in enumerate(Rbins[:-1]):
        deltar = Rbins[i+1] - Rbins[i]
        newbin = seg.max() + 1
        idxbin = np.where((R > rbin) & (R <= rbin + deltar) & (seg==0))
        if i == 3:
            newbin = 0
        seg[idxbin] = newbin
    pf.writeto("sources.fits", seg, clobber=True)


if __name__ == "__main__":
    dataset = "MUSE-DEEP"
    data_dir = os.path.join(context.data_dir, dataset)
    fields = context.fields
    for field in fields:
        os.chdir(os.path.join(data_dir, field))
        img, cube = context.get_field_files(field)
        imgum = make_unsharp_mask(img, redo=False)
        immasked = mask_regions(imgum)
        sexcat = run_sextractor(immasked, redo=True)
        mask_sources(immasked, sexcat, field)
        # remove_sources(field)
        # remove_regions()
        # include_regions()
        # modify_binning(field)

