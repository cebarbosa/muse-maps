#!/usr/bin/env python

"""

Created on 05/02/16

@author: Carlos Eduardo Barbosa

Produces the Voronoi binning.

"""
from __future__ import division, print_function
import os

import numpy as np
from astropy.io import fits, ascii
from astropy.table import Table

from vorbin.voronoi_2d_binning import voronoi_2d_binning

import context
from geomfov import calc_geom
from misc import array_from_header

def collapse_cube(cubename, outfile, redo=False, wmin=5590, wmax=5680):
    """ Collapse a MUSE data cube to produce a white-light image and a
    noise image.

    The noise is estimated with the same definition of the DER_SNR algorithm.

    Input Parameters
    ----------------
    cubename : str
        Name of the MUSE data cube

    outfile : str
        Name of the output file

    redo : bool (optional)
        Redo calculations in case the oufile exists.
    """
    if os.path.exists(outfile) and not redo:
        return
    data = fits.getdata(cubename, 1)
    var = fits.getdata(cubename, 2)
    h0 = fits.getheader(cubename, 0)
    h = fits.getheader(cubename, 1)
    h2 = fits.getheader(cubename, 2)
    wave = array_from_header(cubename)
    idx = np.where((wave <= wmax) & (wave >= wmin))[0]
    h["NAXIS"] = 2
    h2["NAXIS"] = 2
    del_keys = ["NAXIS3", "CTYPE3", "CUNIT3", "CD3_3", "CRPIX3", "CRVAL3",
                "CRDER3", "CD1_3", "CD2_3", "CD3_1", "CD3_2"]
    for key in del_keys:
        del h2[key]
        del h[key]
    print("Starting collapsing process...")
    newdata = np.nanmean(data[idx,:,:], axis=0)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.* data - \
           np.roll(data, 2, axis=0) - np.roll(data, -2, axis=0)), \
           axis=0)
    # noise = np.nanmean(np.sqrt(var[idx,:,:]), axis=0)
    hdu0 = fits.PrimaryHDU()
    hdu0.header = h0
    hdu = fits.ImageHDU(newdata, h)
    hdu2 = fits.ImageHDU(noise, h2)
    hdulist = fits.HDUList([hdu0, hdu, hdu2])
    hdulist.writeto(outfile, overwrite=True)
    return

def calc_binning(signal, noise, mask, targetSN, output, redo=False):
    """ Calculates Voronoi bins using only pixels in a mask.

    Input Parameters
    ----------------
    signal : np.array
        Signal image.

    noise : np.array
        Noise image.

    mask : np.array
        Mask for the combination. Excluded pixels are marked witn NaNs.
        Segregation within mask is indicates by different non-NaN values.

    redo : bool
        Redo the work in case the output file already exists.
    output: str
        Name of the output ascii table.

    Output Parameters
    -----------------
    """
    if os.path.exists(output) and not redo:
        return output
    # Preparing position arrays
    ydim, xdim = signal.shape
    x1 = np.arange(1, xdim+1)
    y1 = np.arange(1, ydim+1)
    xx, yy = np.meshgrid(x1, y1)
    #########################################################################
    # Flatten arrays -- required by Voronoi bin
    signal = signal.flatten()
    noise = noise.flatten()
    mask = mask.flatten()
    xx = xx.flatten()
    yy = yy.flatten()
    #########################################################################
    # Masking
    goodpix = np.logical_and(np.logical_and(np.isfinite(mask), noise >=0.1),
                             signal > 0)
    signal = signal[goodpix]
    noise = noise[goodpix]
    segments = mask[goodpix]
    xx = xx[goodpix]
    yy = yy[goodpix]
    #########################################################################
    # Binning separate sources
    newx = np.zeros_like(xx)
    newy = np.zeros_like(yy)
    bins = np.zeros_like(xx)
    di = 0
    deltabin = 0
    sources = np.unique(segments)
    for i,source in enumerate(sources):
        print("Source {0}/{1:}".format(i+1, len(sources)))
        idx = segments == source
        s = signal[idx]
        n = noise[idx]
        x = xx[idx]
        y = yy[idx]
        try:
            binNum, xNode, yNode, xBar, yBar, sn, nPixels, \
            scale = voronoi_2d_binning(x, y, s, n, targetSN, plot=0,
                                       quiet=0, pixelsize=1, cvt=False)
            binNum += 1
        except ValueError:
            binNum = np.ones_like(x)
        binNum += deltabin
        newx[di:len(x)+di] = x
        newy[di:len(y)+di] = y
        bins[di:len(x)+di] = binNum
        deltabin = binNum.max()
        di += len(x)
    ##########################################################################
    table = Table([newx, newy, bins], names=["X_IMAGE", "Y_IMAGE",
                                             "BIN_NUMBER"])
    table.write(output, format="ascii", overwrite=True)
    return

def make_voronoi_image(bintable, img, targetSN, output, redo=False):
    """ Produces an check image for the Voronoi Tesselation.

    Input Parameters
    ----------------
    bintable : str
        Table containing at least three columns with denominations X_IMAGE,
        Y_IMAGE and BIN_NUMBER.

    img : str
        Fits file image name to determine the dimension of the output image.

    targetSN : float
        Indicates the S/N ratio of the input tesselation to determine the
        output file name.

    output: str
        Name of the output image containing the Voronoi tesselation in 2D.

    redo : bool
        Redo the work in case the output file already exists.
    """
    if os.path.exists(output) and not redo:
        return output
    tabledata = Table.read(bintable, format="ascii")
    imgdata = fits.getdata(img)
    binimg = np.zeros_like(imgdata) * np.nan
    # Making binning scheme
    for line in tabledata:
        i, j = int(line["X_IMAGE"]) - 1, int(line["Y_IMAGE"]) - 1
        binimg[j,i] = line["BIN_NUMBER"]
    hdu = fits.PrimaryHDU(binimg)
    hdu.writeto(output, overwrite=True)
    return output

def sort_voronoi2D(voronoi2D, imgname):
    """ Sort Voronoi tesselation to be increasing as a function of the radius"""
    vor = fits.getdata(voronoi2D)
    newvor = np.zeros_like(vor) * np.nan
    geom = calc_geom(voronoi2D, imgname)
    r = np.array(geom["R"]).astype(float)
    idx = np.argsort(r)
    geom = geom[idx]
    for i, line in enumerate(geom):
        idx = np.where(vor == float(line["BIN"]))
        newvor[idx] = i+1
    hdu = fits.PrimaryHDU(newvor)
    hdu.writeto(voronoi2D, overwrite=True)
    return voronoi2D

def combine_spectra(cubename, voronoi2D, targetSN, field, redo=False):
    """ Produces the combined spectra for a given binning file.

    Input Parameters
    ----------------
    cubename : str
        File for the data cube

    voronoi2D : str
        Fits image containing the Voronoi scheme.

    targetSN : float
        Value of the S/N ratio used in the tesselation

    redo : bool
        Redo combination in case the output spec already exists.

    """
    outdir = os.path.join(os.getcwd(), "sn{}".format(targetSN),
                          "combined".format(targetSN))
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    data = fits.getdata(cubename, 1)
    variance = np.sqrt(fits.getdata(cubename, 2))
    wave = array_from_header(cubename)
    vordata = fits.getdata(voronoi2D)
    vordata = np.ma.array(vordata, mask=np.isnan(vordata))
    bins = np.unique(vordata)[:-1]
    for j, bin in enumerate(bins):
        idx, idy = np.where(vordata == bin)
        ncombine = len(idx)
        print("Bin {0} / {1} (ncombine={2})".format(j + 1, bins.size, ncombine))
        output = os.path.join(outdir, "{}_sn{}_{:04d}.fits".format(field,
                              targetSN, int(bin)))
        if os.path.exists(output) and not redo:
            continue
        errs = np.sqrt(np.nanmean(variance[:,idx,idy], axis=1))
        combined = np.nanmean(data[:,idx,idy], axis=1)
        table = Table([wave, combined, errs], names=["wave", "flux", "fluxerr"])
        table.write(output, overwrite=True)
    return

if __name__ == '__main__':
    targetSN = 100
    for obs in context.obs:
        imgname, cubename = context.get_field_files(obs)
        wdir = os.path.split(imgname)[0]
        os.chdir(wdir)
        outdir = os.path.join(wdir, "sn{}".format(targetSN))
        if not os.path.exists(outdir):
            os.mkdir(outdir)
        snimg = os.path.join(wdir, "signal_noise_{}.fits".format(obs))
        collapse_cube(cubename, snimg, redo=False)
        signal = fits.getdata(snimg, 1)
        noise = fits.getdata(snimg, 2)
        mask = fits.getdata("mask_{}.fits".format(obs))
        outtable = os.path.join(outdir, "voronoi_table_{}.txt".format(obs,
                                                                   targetSN))
        bintable = calc_binning(signal, noise, mask, targetSN,
                                outtable, redo=False)
        vor2D = os.path.join(outdir, "voronoi2d_{}.fits".format(obs,
                                                                  targetSN))
        voronoi2D = make_voronoi_image(bintable, imgname, targetSN,
                                       vor2D, redo=False)
        combine_spectra(cubename, voronoi2D, targetSN, obs, redo=False)