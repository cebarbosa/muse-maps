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

import context
from geomfov import get_geom
from mapplot import PlotVoronoiMaps

def make_tables(key="lick", targetSN=70, nsim=200, dataset="MUSE-DEEP",
                redo=False, sigma=None):
    """ Produces tables for maps of individual indices. """
    # Reading name of the indices
    indexfile = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', "tables", "spindex_CS.dat"))
    # spindex = np.loadtxt(indexfile, usecols=(8,), dtype=str)
    units_bin = np.loadtxt(indexfile, usecols=(7,))
    units = np.where(units_bin, u.mag, u.AA)
    sigma_str = "" if sigma is None else "_sigma{}".format(sigma)
    for field in context.fields[::-1]:
        wdir = os.path.join(context.data_dir, dataset, field )
        output = os.path.join(wdir, "table_{}_{}_sn{}_nsim{}{}.fits".format(key,
                              field, targetSN, nsim, sigma_str))
        if os.path.exists(output) and not redo:
            continue
        geom = get_geom(field, targetSN)
        lick_dir = os.path.join(wdir, "lick" )
        licktables = ["{}_sn{}_{}_nsim{}{}.fits".format(field, targetSN,
                     _, nsim, sigma_str) for _ in geom["BIN"]]
        tables, idx = [], []
        for i, table in enumerate(licktables):
            datafile = os.path.join(lick_dir, table)
            if not os.path.exists(datafile):
                continue
            else:
                idx.append(i)
            data = Table.read(datafile)
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
        if not tables:
            continue
        table = vstack(tables)
        ########################################################################
        # Setting units
        units2x = map(list, zip(units, units))
        units = [item for sublist in units2x for item in sublist]
        for unit, col in zip(units, table.colnames):
            table[col] = table[col] * unit
        ########################################################################
        table = hstack([geom[idx], table])
        table.write(output, format="fits", overwrite=True)

def lick_maps_individual(key="lick", targetSN=70, dataset="MUSE-DEEP",
                         nsim=200, sigma=None):
    """ Produces maps of Lick indices individually. """
    sigma_str = "" if sigma is None else "_sigma{}".format(sigma)
    tables = []
    for field in context.fields:
        tables.append(os.path.join(context.data_dir, dataset, field,
            "table_{}_{}_sn{}_nsim{}{}.fits".format(key, field, targetSN,
                                                    nsim, sigma_str)))
    idx = [i for i,_ in enumerate(tables) if os.path.exists(_)]
    fields = [context.fields[i] for i in idx]
    tables = [tables[i] for i in idx]
    data = [Table.read(_) for _ in tables]
    imin, imax = 6, 32
    ############################################################################
    # Setting columns to be used
    indexfile = os.path.abspath(os.path.join(
                os.path.dirname(__file__), '..', "tables", "spindex_CS.dat"))
    columns = np.loadtxt(indexfile, usecols=(8,), dtype=str)[imin:imax]
    ###########################################################################
    # Calculate limits and correct labels
    lims, labels = [], []
    for col in columns:
        a = np.concatenate([data[i][col] for i in idx])
        q1, q2 = np.percentile(a[np.isfinite(a)], [15, 95])
        lims.append([q1, q2])
        has_muse = True if "_muse" in col else False
        label = col.replace("_muse", "")
        label = "{}$".format(label.replace("_", "$_")) if "_" in label else \
                             label
        label = label.replace("_beta", "\\beta")
        label = label.replace("_D", " D")
        label = "{}$^{{\\rm MUSE}}$".format(label) if has_muse else label
        labels.append(label)
    ###########################################################################
    cmaps = len(columns) * ["YlOrBr"]
    units_bin = np.loadtxt(indexfile, usecols=(7,))
    units = np.where(units_bin, "mag", "\AA")[imin:imax]
    cb_fmts = ["%.2f" if unit =="\AA" else "%.3f" for unit in units]
    labels = ["{0} ({1})".format(x,y) for x,y in zip(labels, units)]
    pvm = PlotVoronoiMaps(data, columns, labels=labels, lims=lims,
                          cmaps=cmaps, cb_fmts=cb_fmts)
    pvm.plot(sigma=sigma)
    return
if __name__ == "__main__":
    sigmas = [None, 300]
    for sigma in sigmas:
        licktable = make_tables(redo=False, sigma=sigma)
        lick_maps_individual(sigma=sigma)


