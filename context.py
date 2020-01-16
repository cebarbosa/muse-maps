# -*- coding: utf-8 -*-
""" 

Created on 23/11/17

Author : Carlos Eduardo Barbosa

Project definitions

"""
import os
import getpass

import matplotlib.pyplot as plt
from astropy.coordinates import SkyCoord
import astropy.units as u
from dustmaps.config import config
from dustmaps import sfd

if getpass.getuser() == "kadu":
    home = "/home/kadu/Dropbox/hydraimf"
else:
    home = "/sto/home/cebarbosa/hydraimf"

data_dir = os.path.join(home, "data")

config['data_dir'] = os.path.join(data_dir, "dustmaps")
if not os.path.exists(config["data_dir"]): # Just to run once in my example
    sfd.fetch() # Specific for Schlafy and Finkbeiner (2011), which is an
    # updated version of the popular Schlegel, Finkbeiner & Davis (1998) maps

fields = ["fieldA", "fieldB", "fieldC", "fieldD"]

# Constants
D = 50.7 # Distance to the center of the Hydra I cluster in Mpc
DL = 55.5# Luminosity distance
velscale = 30. # Set velocity scale for pPXF related routines
V = 3800 # km/s
w1 = 4500
w2 = 10000

# Properties of the system
ra0 = 159.178471651 * u.degree
dec0 = -27.5281283035 * u.degree

# Get color excess
coords = SkyCoord(ra0, dec0)
sfq = sfd.SFDQuery()
ebv = sfq(coords)
Rv = 3.1  # Constant in our galaxy
Av = ebv * Rv

# VHELIO - radial velocities of the fields, have to be added from the
# observed velocities.
vhelio = {"fieldA" : 24.77, "fieldB" : 21.26, "fieldC" : 20.80,
          "fieldD" : 19.09} # km / s

# Matplotlib settings
plt.style.context("seaborn-paper")
plt.rcParams["text.usetex"] = True
plt.rcParams["font.family"] = "serif"
plt.rcParams['font.serif'] = 'Computer Modern'
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.minor.visible"] = True
plt.rcParams["ytick.minor.visible"] = True
plt.rcParams["xtick.top"] = True
plt.rcParams["ytick.right"] = True

def get_field_files(field, dataset="MUSE"):
    """ Returns the names of the image and cube associated with a given
    field. """
    if dataset == "MUSE-DEEP":
        wdir = os.path.join(home, "data/MUSE-DEEP", field)
        if field == "fieldA":
            img = "ADP.2017-03-27T12:49:43.628.fits"
            cube = "ADP.2017-03-27T12:49:43.627.fits"
        elif field == "fieldB":
            img = "ADP.2017-03-27T12:49:43.652.fits"
            cube = "ADP.2017-03-27T12:49:43.651.fits"
        elif field == "fieldC":
            img = "ADP.2017-03-27T12:49:43.644.fits"
            cube = "ADP.2017-03-27T12:49:43.643.fits"
        elif field == "fieldD":
            img = "ADP.2017-03-27T12:49:43.636.fits"
            cube = "ADP.2017-03-27T12:49:43.635.fits"
        return os.path.join(wdir, img), os.path.join(wdir, cube)
    elif dataset=="MUSE":
        wdir = os.path.join(home, "data/MUSE", "combined", field)
        img = os.path.join(wdir, "NGC3311_{}_IMAGE_COMBINED.fits".format(field))
        cube = os.path.join(wdir, "NGC3311_{}_DATACUBE_COMBINED.fits".format(
            field))
        return img, cube
    else:
        raise ValueError("Data set name not defined: {}".format(dataset))

def get_data_dir(dataset):
    if dataset == "MUSE-DEEP":
        return os.path.join(home, "data/MUSE-DEEP")
    elif dataset == "MUSE":
        return os.path.join(home, "data/MUSE/combined" )

# Emission lines used in the projects
def get_emission_lines():
    """ Returns dictionaries containing the emission lines to be used. """
    lines = (("Hbeta_4861", 4861.333), ("OIII_4959", 4958.91),
             ("OIII_5007", 5006.84), ("NII_6550", 6549.86),
             ("Halpha_6565", 6564.61), ("NII_6585", 6585.27),
             ("SII_6718", 6718.29), ("SII_6733", 6732.67))
    return lines
