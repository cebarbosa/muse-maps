# -*- coding: utf-8 -*-
""" 

Created on 09/03/18

Author : Carlos Eduardo Barbosa

Clean telluric lines from spectra using molecfit.

"""
from __future__ import print_function, division

import os
import pickle

from astropy.io import fits

import context

def run(redo=False):
    """ Main routine to process the data. """
    for field in context.fields:
        prepare_input(field)
        # run molecfit
        # apply models
        # make images

def prepare_input():
    """ Prepare fits tables according to specifications of molecfit. """
    pass

if __name__ == "__main__":
    run(redo=True)