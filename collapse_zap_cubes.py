# -*- coding: utf-8 -*-
""" 

Created on 30/10/18

Author : Carlos Eduardo Barbosa

Produces collapsed images of the data cubes processed with ZAP.

"""
from __future__ import print_function, division

import os

from astropy.table import Table

import context
from make_voronoi_binning import collapse_cube

if __name__ == "__main__":
    cubes = Table.read(os.path.join(context.data_dir,
                                  "MUSE/tables/zap_table.fits"))["out_cube"]
    wdir = os.path.join(context.data_dir, "MUSE/zap")
    os.chdir(wdir)
    for cube in cubes:
        if not os.path.exists(cube):
            continue
        print(cube)
        imgfile = cube.replace(".fits", "_img.fits")
        collapse_cube(cube, imgfile, redo=False)