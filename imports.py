# %load imports.py
import os
import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm
import glob
import yaml

import param
import panel as pn
import holoviews as hv
from holoviews import opts
hv.extension('bokeh', 'matplotlib')
from bokeh.io import export_png, export_svgs


opts.defaults(opts.Scatter(width=1000, height=300),
              opts.Histogram(width=1000, height=300),
              opts.Image(width=1000, height=300),
              opts.Curve(width=1000, height=300),
              opts.Points(width=1000, height=300))


%pylab inline
#from matplotlib.colors import LogNorm
%config InlineBackend.figure_format ='retina'

rcParams['figure.figsize'] = (13.0, 6.)

from scipy.optimize import curve_fit
from scipy.stats import norm
        
def get_data_pd(fname: str) -> pd.DataFrame:
    try:
        with h5py.File(fname, 'r') as f:
            rawNr  = f['raw/trigger nr'][:]
            rawTof = f['raw/tof'][:]*1e6
            rawTot = f['raw/tot'][:]
            rawX   = f['raw/x'][:]
            rawY   = f['raw/y'][:]
            centNr = f['centroided/trigger nr'][:]
            centTof= f['centroided/tof'][:]*1e6
            centTot= f['centroided/tot max'][:]
            centY  = f['centroided/y'][:]
            centX  = f['centroided/x'][:]
            
        raw_data = pd.DataFrame(np.column_stack((rawNr, rawTof, rawTot, rawX, rawY)),
                       columns=('nr', 'tof', 'tot', 'x', 'y'))
        cent_data = pd.DataFrame(np.column_stack((centNr, centTof, centTot, centX, centY)),
                       columns=('nr', 'tof', 'tot', 'x', 'y'))
        return raw_data, cent_data
    except:
        print(f'key "{keys}" not known or file "{fname}" not existing')

        
def gauss_fwhm(x, *p):
    A, mu, fwhm = p
    return A * np.exp(-(x - mu) ** 2 / (2. * (fwhm ** 2)/(4*2*np.log(2))))