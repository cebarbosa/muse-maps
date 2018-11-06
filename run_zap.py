# -*- coding: utf-8 -*-
""" 

Created on 24/10/18

Author : Carlos Eduardo Barbosa

Run ZAP over MUSE dataset.

"""
from __future__ import print_function, division

import os

from astropy.io import fits
from astropy.table import Table
import zap

import context

def prepare_zap_input():
    """ Make table with corresponding science and sky cubes. """
    output = os.path.join(context.data_dir, "MUSE/tables/zap_table.fits")
    if os.path.exists(output):
        table = Table.read(output)
        return table
    datacubes = []
    for dir_ in ["sci", "sky"]:
        data_dir = os.path.join(context.data_dir, "MUSE", dir_)
        fnames = [os.path.join(data_dir,_) for _ in os.listdir(data_dir) if
                  _.endswith("fits") and _.startswith("ADP")]
        datacubes.append([_ for _ in fnames if fits.getval(_, "NAXIS",
                                                           ext=1)==3])
    scicubes, skycubes = datacubes
    skyobj = [fits.getval(_, "OBJECT") for _ in skycubes]
    corresponding_sky, outcubes = [], []
    for scicube in scicubes:
        obj = fits.getval(scicube, "OBJECT")
        s = "SKY_for_{}".format(obj)
        idx = skyobj.index(s)
        corresponding_sky.append(skycubes[idx])
        outcubes.append("{}.fits".format(obj))
    table = Table([scicubes, corresponding_sky, outcubes],
                  names=["sci_cube", "sky_cube", "out_cube"])
    table.write(output, overwrite=True)
    return table



if __name__ == "__main__":
    zap_dir = os.path.join(context.data_dir, "MUSE/zap")
    if not os.path.exists(zap_dir):
        os.mkdir(zap_dir)
    os.chdir(zap_dir)
    table = prepare_zap_input()
    for field in table:
        output = os.path.join(os.getcwd(), "{}.fits".format(field["out_cube"]))
        if os.path.exists(output):
            continue
        mask = os.path.join(context.data_dir, "sky", "skymask_{}.fits".format(
            field["out_cube"].replace("NGC3311_", "")))
        extSVD = zap.SVDoutput(field["sky_cube"], mask=mask)
        zap.process(field["sci_cube"], outcubefits=output,
                    extSVD=extSVD)
        break