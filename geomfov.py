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
from astropy.table import Table

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

def calc_geom(binfile, imgfile):
    """Calculate the location of bins for a given target S/N and field. """
    binimg = fits.getdata(binfile)
    extent = calc_extent(imgfile)
    #  TODO: check if geometry is correct in MUSE-DEEP data set
    # extent = offset_extent(extent, field)
    ydim, xdim = binimg.shape
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
    specs = np.array(["{0:04d}".format(x) for x in bins])
    table = Table(data=[specs, xcen, ycen, radius, pa], names=["BIN", "X",
                                                               "Y", "R", "PA"])
    return table

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

if __name__ == "__main__":
    pass