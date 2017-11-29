#!/usr/bin/env python

"""

Created on 05/02/16

@author: Carlos Eduardo Barbosa

Produces the Voronoi binning.

"""
from __future__ import division, print_function
import os

import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

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

def calc_binning(w1, w2, targetSN):
    """ Performs binning for a given MUSE collpsed cube. """
    fits = "collapsed_w{0}_{1}.fits".format(w1, w2)
    output = "voronoi_2d_sn{0}_w{1}_{2}.txt".format(targetSN, w1, w2)
    badpix = fits.getdata(mask_file)
    signal = fits.getdata(fits, 0)
    noise = fits.getdata(fits, 1)
    segments = fits.getdata(sources_file).astype(float)
    ##########################################################################
    # Preparing position arrays
    ydim, xdim = signal.shape
    x1 = np.arange(1, xdim+1)
    y1 = np.arange(1, ydim+1)
    xx, yy = np.meshgrid(x1, y1)
    ##########################################################################
    # Preparing arrays for masking
    regions = badpix == 1. # Bad pixels from ds9 regions
    signal[regions] = np.nan
    #########################################################################
    # Flatten arrays
    signal = signal.flatten()
    noise = noise.flatten()
    segments = segments.flatten()
    xx = xx.flatten()
    yy = yy.flatten()
    #########################################################################
    # Masking
    goodpix = np.logical_and(np.isfinite(signal), np.isfinite(noise))
    signal = signal[goodpix]
    noise = noise[goodpix]
    segments = segments[goodpix]
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
    if os.path.exists("avoid_bins_sn{0}.txt".format(targetSN)):
        avoid = np.loadtxt("avoid_bins_sn{0}.txt".format(targetSN))
    else:
        avoid = []
    for i,source in enumerate(sources[::-1]):
        print("Source {0}/{1:}".format(i+1, len(sources)))
        idx = segments == source
        s = signal[idx]
        n = noise[idx]
        x = xx[idx]
        y = yy[idx]
        if source in avoid:
            binNum = np.ones_like(x)
        else:
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
    # Save to a text file the initial coordinates of each pixel together
    # with the corresponding bin number computed by this procedure.
    # binNum uniquely specifies the bins and for this reason it is the only
    # number required for any subsequent calculation on the bins.
    #
    np.savetxt(output, np.column_stack([newx, newy, bins]),
               fmt=b'%10.6f %10.6f %8i')
    return

def make_voronoi_image(w1, w2, targetSN):
    """ Produces an check image for the Voronoi Tesselation. """
    voronoi_file = "voronoi_2d_sn{0}_w{1}_{2}.txt".format(targetSN, w1, w2)
    fits = "collapsed_w{0}_{1}.fits".format(w1, w2)
    xpixel, ypixel, binnum = np.loadtxt(voronoi_file).T
    img = fits.getdata(fits)
    binimg = np.zeros_like(img)
    binimg[:] = np.nan
    ydim, xdim = img.shape
    # Making binning scheme
    for x,y,value in zip(ypixel.astype(int), xpixel.astype(int), binnum):
        binimg[x-1,y-1] = value
    fits.writeto("voronoi_sn{0}_w{1}_{2}.fits".format(targetSN, w1, w2),
               binimg, clobber=True)
    plt.savefig("figs/voronoi_sn{0}_w{1}_{2}.png".format(targetSN, w1, w2))
    plt.clf()
    return

def combine_spectra(w1, w2, targetSN):
    """ Produces the combined spectra for a given binning file. """
    datacube = [x for x in os.listdir(".") if "DATACUBE_FINAL" in x][0]
    voronoi = "voronoi_sn{0}_w{1}_{2}.fits".format(targetSN, w1, w2)
    data = fits.getdata(datacube, 1)
    #########################################################################
    # Adapt header
    h = fits.getheader(datacube, 1)
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
    vordata = pf.getdata(voronoi)
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
    hdulist.writeto("binned_sn{0}.fits".format(targetSN), clobber=True)
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
        # plt.imshow(signal / noise, origin="bottom", vmin=0, vmax=2)
        # plt.colorbar()
        # plt.show()
        # break
        # calc_binning(img, targetSN)
        # make_voronoi_image(w1, w2, targetSN)
        # combine_spectra(w1, w2, targetSN)


if __name__ == '__main__':
    fields = context.fields
    run(fields, targetSN=70, dataset="MUSE-DEEP")
