# %load imports.py
import glob
import os

import h5py
import holoviews as hv
import numpy as np
import pandas as pd
import panel as pn
import param
import yaml
from holoviews import opts
from scipy.constants import c, physical_constants
from tqdm import tqdm

hv.extension("bokeh", "matplotlib")
from bokeh.io import export_png, export_svgs

opts.defaults(
    opts.Scatter(width=1000, height=300, tools=["hover"]),
    opts.Histogram(width=1000, height=300, tools=["hover"]),
    opts.Image(width=1000, height=300, tools=["hover"]),
    opts.Curve(width=1000, height=300, tools=["hover"]),
    opts.Points(width=1000, height=300, tools=["hover"]),
)


%pylab inline
# from matplotlib.colors import LogNorm
%config InlineBackend.figure_format ='retina'

rcParams["figure.figsize"] = (13.0, 6.0)

from scipy.optimize import curve_fit
from scipy.stats import norm

from utils import *

get_x_axis_from_bins = lambda x_bins: 0.5 * (x_bins[1:] + x_bins[:-1])
file_title = lambda x: os.path.basename(x).rstrip(".hdf5")

with open("runs.yaml") as f:
    runNrs = yaml.safe_load(f)