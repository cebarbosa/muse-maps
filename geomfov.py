# -*- coding: utf-8 -*-
""" 

Created on 04/12/17

Author : Carlos Eduardo Barbosa

Routines related to the geometry of the FoV of the observations.

"""
from __future__ import print_function, division

import os

import numpy as np
from astropy.io import fits

import context
from misc import array_from_header

def calc_extent(image, extension=1):
    h = fits.getheader(image)
    ra = array_from_header(image, axis=1, extension=extension)
    dec = array_from_header(image, axis=2, extension=extension)
    # Ofset to the center of NGC 3311
    ra -= context.ra0
    dec -= context.dec0
    # Convert to radians
    X = context.D * 1000 * np.deg2rad(ra)
    Y = context.D * 1000 * np.deg2rad(dec)
    # Scale to the distance of the cluster
    extent = np.array([X[0], X[-1], Y[0], Y[-1]])
    return extent

def calc_geom(sn, field):
    """Calculate the location of bins for a given target S/N and field. """
    binsfile = "voronoi_sn{0}_w4500_5500.fits".format(sn)
    binimg = fits.getdata(os.path.join(context.data_dir, "combined_{}".format(
             field), binsfile))
    ydim, xdim = binimg.shape
    intens = os.path.join(os.path.join(context.data_dir, "combined_{}".format(
        field),
                                "collapsed_w4500_5500.fits"))
    extent = calc_extent(intens)
    #  TODO: check if geometry is correct in MUSE-DEEP data set
    # extent = offset_extent(extent, field)
    x = np.linspace(extent[0], extent[1], xdim)
    y = np.linspace(extent[2], extent[3], ydim)
    xx, yy = np.meshgrid(x, y)
    binimg = np.ma.array(binimg, mask=~np.isfinite(binimg))
    bins = np.arange(binimg.min(), binimg.max()+1)
    xcen = np.zeros_like(bins)
    ycen = np.zeros_like(xcen)
    bins = bins.astype(int)
    for i, bin in enumerate(bins):
        idx = np.where(binimg == bin)
        xcen[i] = np.mean(xx[idx])
        ycen[i] = np.mean(yy[idx])
    radius = np.sqrt(xcen**2 + ycen**2)
    pa = np.rad2deg(np.arctan2(xcen, ycen))
    # Converting to strings
    xcen = ["{0:.5f}".format(x) for x in xcen]
    ycen = ["{0:.5f}".format(x) for x in ycen]
    radius = ["{0:.5f}".format(x) for x in radius]
    pa = ["{0:.5f}".format(x) for x in pa]
    specs = np.array(["{0}_bin{1:04d}".format(field, x) for x in bins])
    return np.column_stack((specs, xcen, ycen, radius, pa))