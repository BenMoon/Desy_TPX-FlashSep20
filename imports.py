# %load imports.py
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


def get_data_pd(fname: str) -> pd.DataFrame:
    try:
        with h5py.File(fname, "r") as f:
            rawNr = f["raw/trigger nr"][:]
            rawTof = f["raw/tof"][:] * 1e6
            rawTot = f["raw/tot"][:]
            rawX = f["raw/x"][:]
            rawY = f["raw/y"][:]
            centNr = f["centroided/trigger nr"][:]
            centTof = f["centroided/tof"][:] * 1e6
            centTot = f["centroided/tot max"][:]
            centY = f["centroided/y"][:]
            centX = f["centroided/x"][:]

        raw_data = pd.DataFrame(
            np.column_stack((rawNr, rawTof, rawTot, rawX, rawY)),
            columns=("nr", "tof", "tot", "x", "y"),
        )
        cent_data = pd.DataFrame(
            np.column_stack((centNr, centTof, centTot, centX, centY)),
            columns=("nr", "tof", "tot", "x", "y"),
        )
        return raw_data, cent_data
    except:
        print(f'key "{keys}" not known or file "{fname}" not existing')


def gauss_fwhm(x, *p):
    A, mu, fwhm = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * (fwhm ** 2) / (4 * 2 * np.log(2))))


def find_peaks_in_microbunch(
    data: pd.DataFrame, nr_peaks: int = 4, dt: float = 10, offset: float = 0
) -> list:
    """find first peak in micro-bunch"""
    peaks = []
    for i in range(nr_peaks):
        mask = np.logical_and(
            data["tof"] > (offset + i * dt), data["tof"] < (offset + i * dt + 1)
        )
        x_hist, x_edges = np.histogram(data["tof"][mask], bins=1_000)
        x = (x_edges[:-1] + x_edges[1:]) * 0.5
        popt, pcov = curve_fit(
            gauss_fwhm, x, x_hist, p0=[x_hist.max(), x[x_hist.argmax()], 0.05]
        )
        peaks.append(popt[1])
    return peaks


def shift_microbunch_pulses(
    data: pd.DataFrame, nr_peaks: int = 4, dt: float = 10, offset: float = 0
) -> pd.DataFrame:
    """Fold consecutive micro-bunch pulses back to first"""
    peaks = find_peaks_in_microbunch(data, nr_peaks, dt, offset)

    # shift bunches
    for i in range(1, nr_peaks):
        mask = np.logical_and(
            data["tof"] >= offset + i * dt, data["tof"] < offset + (i + 1) * dt
        )
        data["tof"][mask] -= peaks[i] - peaks[0]

    return data


def radial_profile(data: np.array, center: tuple) -> np.array:
    y, x = np.indices(data.shape)
    r = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

get_x_axis_from_bins = lambda x_bins: 0.5 * (x_bins[1:] + x_bins[:-1])
file_title = lambda x: os.path.basename(x).rstrip(".hdf5")

with open("runs.yaml") as f:
    runNrs = yaml.safe_load(f)