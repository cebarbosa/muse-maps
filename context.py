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

def get_field_files(field):
    """ Returns the names of the image and cube associated with a given
    field. """
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
    return img, cube
