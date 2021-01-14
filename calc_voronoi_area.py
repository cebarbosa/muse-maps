import os

import astropy.units as u
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt

import context

if __name__ == "__main__":
    wdir = os.path.join(context.data_dir, "MUSE/sn100")
    filenames = [_ for _ in os.listdir(wdir) if _.endswith(".fits")]
    voronoi = []
    vmax = 0
    for i, fname in enumerate(filenames):
        vdata = fits.getdata(os.path.join(wdir, fname)).flatten()
        vdata = vdata[np.isfinite(vdata)]
        voronoi.append(vdata + vmax)
        nbins = len(np.unique(vdata))
        vmax += nbins
    voronoi = np.concatenate(voronoi)
    hist, bins = np.histogram(voronoi, bins=np.unique(vdata))
    n = np.median(hist)
    ps = 0.2
    r = np.sqrt(n * ps**2 / np.pi)
    d = 2 * r
    plt.hist(hist, bins=bins)
    arcsec2kpc = 0.059
    print(d, d * arcsec2kpc)
    plt.show()