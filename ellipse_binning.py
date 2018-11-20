# -*- coding: utf-8 -*-
""" 

Created on 19/11/18

Author : Carlos Eduardo Barbosa

Run ellipse into image to determine regions to combine spectra.

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.modeling.models import Ellipse2D
import matplotlib.pyplot as plt
from photutils.isophote import Ellipse
from photutils.isophote import EllipseGeometry, build_ellipse_model

import context

def make_masked_img(img, maskfile=None):
    """ Read image and produces a masked array. """
    maskfile = "halo_only.fits" if maskfile is None else maskfile
    mask = fits.getdata(maskfile)
    mask = np.where(np.isnan(mask), 1, 0)
    data = fits.getdata(img)
    norm = np.nanmedian(data)
    data /= norm
    data = np.ma.array(data, mask=mask)
    return data, norm

def run_ellipse(img, redo=False):
    """ Run ellipse fitting using NGC 3311 MUSE images. """
    # Reading data and mask
    outfile = "ellipse.txt"
    if os.path.exists(outfile) and not redo:
        return
    data = make_masked_img(img)
    # Preparing ellipse fitting

    geometry = EllipseGeometry(x0=213, y0=235, sma=25, eps=0.3,
                               pa=np.deg2rad(-50))
    ellipse = Ellipse(data, geometry)
    isolist = ellipse.fit_image(fflag=0.01, maxsma=200, maxrit=104)
    # isolist = ellipse.fit_image(fflag=0.01, maxsma=20)
    table = isolist.to_table()[1:]
    table.write(outfile, format="ascii", overwrite=True)
    # Producing image
    model_image = build_ellipse_model(data.shape, isolist)
    residual = data - model_image
    fig, (ax1, ax2, ax3) = plt.subplots(figsize=(14, 5), nrows=1, ncols=3)
    fig.subplots_adjust(left=0.04, right=0.98, bottom=0.02, top=0.98)
    ax1.imshow(data, origin='lower')
    ax1.set_title('Data')

    smas = np.linspace(5, 200, 10)
    for sma in smas:
        iso = isolist.get_closest(sma)
        x, y, = iso.sampled_coordinates()
        ax1.plot(x, y, color='C1')
    ax2.imshow(model_image, origin='lower')
    ax2.set_title('Ellipse Model')
    ax3.imshow(residual, origin='lower')
    ax3.set_title('Residual')
    plt.savefig("ellipse.png", dpi=250)
    plt.show()

def make_binning(img, skip=5):
    """ Uses results from  ellipse to make binning for the analysis."""
    table = Table.read("ellipse.txt", format="ascii")
    table = table[table["sma"]>10]
    table = table[::skip]
    data, norm = make_masked_img(img)
    ydim, xdim = data.shape
    x, y = np.meshgrid(np.arange(xdim)+1, np.arange(ydim)+1)
    ##########################################################################
    # Set the value of the geometric parameters for the inner region to
    # ignore the dust lane.
    sma_inn = 20
    idxinn = np.where(table["sma"]<=sma_inn)[0]
    e_inn = table[idxinn]["ellipticity"][-1]
    pa_inn = table[idxinn]["pa"][-1]
    ###########################################################################
    bintable = []
    for i, isophote in enumerate(table):
        ellip = e_inn if isophote["sma"] <= sma_inn else isophote["ellipticity"]
        pa = pa_inn if isophote["sma"] <= sma_inn else isophote["pa"]
        bintable.append([isophote["sma"], ellip, pa])
        e = Ellipse2D(amplitude=1., x_0=213, y_0=235, a=isophote["sma"],
                      b=isophote["sma"] * (1 - ellip),
                      theta=np.deg2rad(pa))
        if i==0:
            binning = e(x,y)
            continue
        outer = e(x,y)
        ring = np.where((outer>0) & (binning==0))
        binning[ring] = i+1
    binning[binning==0] = np.nan
    bintable = np.array(bintable)
    table = Table(bintable, names=["sma", "ellipticity", "pa"])
    binning[data.mask] = np.nan
    hdu0 = fits.PrimaryHDU(binning)
    hdu1 = fits.BinTableHDU(table)
    hdulist = fits.HDUList([hdu0, hdu1])
    hdulist.writeto("ellipse_binning_v0.fits", overwrite=True)

def combine_spectra(cubename, binning, redo=False):
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
    outdir = os.path.join(os.getcwd(), "spec1d_ellipv0")
    if not os.path.exists(outdir):
        os.mkdir(outdir)
    data = fits.getdata(cubename, 1)
    variance = np.sqrt(fits.getdata(cubename, 2))
    h = fits.getheader(cubename, 1)
    ############################################################################
    # Preparing header for output
    hnew = fits.Header()
    hnew["NAXIS"] = 1
    kws = ["CRVAL1", "CD1_1", "CRPIX1", 'CUNIT1', "CTYPE1", "NAXIS1",
           "SIMPLE", "BITPIX", "EXTEND"]
    for kw in kws:
        if kw in h.keys():
            hnew[kw] = h[kw.replace("1", "3")]
    ##########################################################################
    zdim, ydim, xdim = data.shape
    h["NAXIS1"] = zdim
    vordata = fits.getdata(binning)
    vordata = np.ma.array(vordata, mask=np.isnan(vordata))
    bins = np.unique(vordata)[:-1]
    for j, bin in enumerate(bins):
        idx, idy = np.where(vordata == bin)
        ncombine = len(idx)
        print("Bin {0} / {1} (ncombine={2})".format(j + 1, bins.size, ncombine))
        output = os.path.join(outdir, "bin{:04d}.fits".format(int(bin)))
        if os.path.exists(output) and not redo:
            continue
        errs = np.sqrt(np.nanmean(variance[:,idx,idy], axis=1))
        combined = np.nanmean(data[:,idx,idy], axis=1)
        hdu = fits.PrimaryHDU(combined, hnew)
        hdu2 = fits.ImageHDU(errs, hnew)
        hdulist = fits.HDUList([hdu, hdu2])
        hdulist.writeto(output, overwrite=True)
    return

if __name__ == "__main__":
    img, cube = context.get_field_files(context.fields[0])
    wdir, img = os.path.split(img)
    os.chdir(wdir)
    run_ellipse(img, redo=False)
    make_binning(img)
    combine_spectra(cube, "ellipse_binning_v0.fits")