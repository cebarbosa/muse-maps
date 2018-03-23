# -*- coding: utf-8 -*-
""" 

Created on 23/03/18

Author : Carlos Eduardo Barbosa

Produces 2D maps of Lick indices

"""
from __future__ import print_function, division

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                '..')))

import numpy as np
import astropy.units as u
from astropy.table import Table, vstack, hstack
import matplotlib.pyplot as plt

import context
from geomfov import get_geom

def make_tables(key="lick", targetSN=70, nsim=200, dataset="MUSE-DEEP",
                redo=False):
    """ Produces tables for maps of individual indices. """
    # Reading name of the indices
    indexfile = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', "tables", "spindex_CS.dat"))
    # spindex = np.loadtxt(indexfile, usecols=(8,), dtype=str)
    units_bin = np.loadtxt(indexfile, usecols=(7,))
    units = np.where(units_bin, u.mag, u.AA)
    for field in context.fields[::-1]:
        wdir = os.path.join(context.data_dir, dataset, field )
        output = os.path.join(wdir, "table_{}_{}_sn{}_nsim{}.fits".format(key,
                              field, targetSN, nsim))
        if os.path.exists(output) and not redo:
            continue
        geom = get_geom(field, targetSN)
        lick_dir = os.path.join(wdir, "lick" )
        licktables = ["{}_sn{}_{}_nsim{}.fits".format(field, targetSN,
                     _, nsim) for _ in geom["BIN"]]
        tables = []
        for table in licktables:
            data = Table.read(os.path.join(lick_dir, table))
            # Setting headers of the tables
            names = np.array(data["name"].tolist(), dtype="U25")
            nameserr = np.array(["{}_err".format(_) for _ in names],
                                dtype=names.dtype)
            newnames = np.empty(2 * len(names), dtype="U25")
            newnames[0::2] = names
            newnames[1::2] = nameserr
            # Reading data and making new table
            values = data[key]
            errors = data["{}err".format(key)]
            data = np.empty(2 * len(values), dtype=values.dtype)
            data[0::2] = values
            data[1::2] = errors
            newtab = Table(data, names=newnames)
            tables.append(newtab)
        table = vstack(tables)
        ########################################################################
        # Setting units
        units2x = map(list, zip(units, units))
        units = [item for sublist in units2x for item in sublist]
        for unit, col in zip(units, table.colnames):
            table[col] = table[col] * unit
        ########################################################################
        table = hstack([geom, table])
        table.write(output, format="fits", overwrite=True)

if __name__ == "__main__":
    make_tables(redo=False)



