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
import matplotlib.pyplot as plt
from astropy.table import Table

import context
from misc import array_from_header
from voronoi.voronoi_2d_binning import voronoi_2d_binning

def collapse_cube(cubename, outfile, redo=False):
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
    h = fits.getheader(cubename, 1)
    h2 = fits.getheader(cubename, 2)
    h["NAXIS"] = 2
    del h["NAXIS3"]
    h2["NAXIS"] = 2
    del h2["NAXIS3"]
    print("Starting collapsing process...")
    newdata = np.nanmedian(data, axis=0)
    noise = 1.482602 / np.sqrt(6.) * np.nanmedian(np.abs(2.* data - \
           np.roll(data, 2, axis=0) - np.roll(data, -2, axis=0)), \
           axis=0)
    hdu = fits.PrimaryHDU(newdata, h)
    hdu2 = fits.ImageHDU(noise, h2)
    hdulist = fits.HDUList([hdu, hdu2])
    hdulist.writeto(outfile, overwrite=True)
    return

def calc_binning(signal, noise, mask, targetSN, redo=False):
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

    Output Parameters
    -----------------
    str
        Name of the output ascii table.
    """
    output = "voronoi_table_sn{}.txt".format(targetSN)
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
    for i,source in enumerate(sources[::-1]):
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
    return output

def make_voronoi_image(bintable, img, targetSN, redo=False):
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

    redo : bool
        Redo the work in case the output file already exists.

    Output Parameters:
        str
        Name of the output image containing the Voronoi tesselation in 2D.
    """
    output = "voronoi2d_sn{}.fits".format(targetSN)
    if os.path.exists(output) and not redo:
        return output
    tabledata = ascii.read(bintable)
    imgdata = fits.getdata(img)
    binimg = np.zeros_like(imgdata) * np.nan
    # Making binning scheme
    for line in tabledata:
        i, j = int(line["X_IMAGE"]) - 1, int(line["Y_IMAGE"]) - 1
        binimg[j,i] = line["BIN_NUMBER"]
    hdu = fits.PrimaryHDU(binimg)
    hdu.writeto(output, overwrite=True)
    return output

def combine_spectra(cubename, voronoi2D, targetSN):
    """ Produces the combined spectra for a given binning file.

    Input Parameters
    ----------------
    cubename : str
        File for the data cube
    voronoi2D : str
        Fits image containing the Voronoi scheme.


    """
    data = fits.getdata(cubename, 1)
    #########################################################################
    # Adapt header
    h = fits.getheader(cubename, 1)
    h["NAXIS"] = 2
    kws = ["CRVAL1", "CD1_1", "CRPIX1", 'CUNIT1', "CTYPE1"]
    h['CUNIT1'] = h['CUNIT3']
    for kw in kws:
        if kw in h.keys():
            h[kw] = h[kw.replace("1", "3")]
            del h[kw.replace("1", "3")]
    del h["NAXIS3"]
    h["CRVAL2"] = 1
    h["CD2_2"] = 1
    h["CRPIX2"] = 1
    h["CTYPE2"] = "BIN NUMBER"
    ##########################################################################
    zdim, ydim, xdim = data.shape
    h["NAXIS1"] = zdim
    vordata = fits.getdata(voronoi2D)
    vordata = np.ma.array(vordata, mask=np.isnan(vordata))
    bins = np.unique(vordata)[:-1]
    h["NAXIS2"] = bins.size
    combined = np.zeros((bins.size,zdim))
    for j, bin in enumerate(bins):
        print("Bin {0} / {1}".format(j+1, bins.size))
        idx, idy = np.where(vordata == bin)
        specs = data[:,idx,idy]
        combined[j,:] = np.nanmean(specs, axis=1)
    print("Writing to disk...")
    hdu = fits.PrimaryHDU(combined, h)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto("binned_sn{0}.fits".format(targetSN), overwrite=True)
    print("Done!")
    return

def run(fields, targetSN=70, dataset="MUSE"):
    for field in fields:
        print(field)
        os.chdir(os.path.join(context.data_dir, dataset, field))
        imgname, cubename = context.get_field_files(field)
        newimg = "whitelamp.fits"
        collapse_cube(cubename, newimg, redo=False)
        signal = fits.getdata(newimg, 0)
        noise = fits.getdata(newimg, 1)
        mask = fits.getdata("simple_binning.fits")
        bintable = calc_binning(signal, noise, mask, targetSN, redo=False)
        voronoi2D = make_voronoi_image(bintable, newimg, targetSN, redo=False)
        combine_spectra(cubename, voronoi2D, targetSN)


if __name__ == '__main__':
    fields = context.fields
    run(fields, targetSN=70, dataset="MUSE-DEEP")