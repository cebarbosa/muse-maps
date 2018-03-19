# -*- coding: utf-8 -*-
""" 

Created on 15/03/18

Author : Carlos Eduardo Barbosa

"""
from __future__ import print_function, division

import os
import pickle

import matplotlib.pyplot as plt

import context
from basket.lick.lick import Lick

import context
from run_ppxf import pPXF

def run_lick(w1, w2, targetSN, dataset="MUSE-DEEP", redo=False, velscale=None):
    """ Calculates Lick indices and uncertainties based on pPXF fitting. """
    if velscale is None:
        velscale = context.velscale
    for field in context.fields:
        wdir = os.path.join(context.data_dir, dataset, field)
        data_dir = os.path.join(wdir, "ppxf_vel{}_w{}_{}_sn{}".format(int(
            velscale), w1, w2, targetSN))
        pkls = sorted([_ for _ in os.listdir(data_dir) if _.endswith(".pkl")])
        for pkl in pkls:
            with open(os.path.join(data_dir, pkl)) as f:
                pp = pickle.load(f)
        lick = Lick(pp.table["wave"], pp.table["flux"] - pp.table["emission"],
                    bands0,
                    vel=0, dw=2.)

        raw_input()
        plt.show()



if __name__ == "__main__":
    targetSN = 70
    w1 = 4500
    w2 = 10000
    run_lick(w1, w2, targetSN)
