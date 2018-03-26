# -*- coding: utf-8 -*-
""" 

Created on 23/11/17

Author : Carlos Eduardo Barbosa

Project definitions

"""
import os

home = "/home/kadu/Dropbox/hydraimf"
data_dir = os.path.join(home, "data")

fields = ["fieldA", "fieldB", "fieldC", "fieldD"]

# Constants
D = 50.7 # Distance to the center of the Hydra I cluster in Mpc
DL = 55.5# Luminosity distance
velscale = 30. # Set velocity scale for pPXF related routines

# Properties of the system
ra0 = 159.178471651
dec0 = -27.5281283035

# VHELIO - radial velocities of the fields, have to be added from the
# observed velocities.
vhelio = {"fieldA" : 24.77, "fieldB" : 21.26, "fieldC" : 20.80,
          "fieldD" : 19.09} # km / s

def get_field_files(field, dataset="MUSE-DEEP"):
    """ Returns the names of the image and cube associated with a given
    field. """
    if dataset == "MUSE-DEEP":
        wdir = os.path.join(data_dir, "MUSE-DEEP", field)
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
        raise(NotImplementedError)
    else:
        raise ValueError("Data set name not defined: {}".format(dataset))

# Emission lines used in the projects
def get_emission_lines():
    """ Returns dictionaries containing the emission lines to be used. """
    lines = (("Hbeta_4861", 4861.333), ("OIII_4959", 4958.91),
             ("OIII_5007", 5006.84), ("NII_6550", 6549.86),
             ("Halpha_6565", 6564.61), ("NII_6585", 6585.27),
             ("SII_6718", 6718.29), ("SII_6733", 6732.67))
    return lines
