# this file contains a collections of functions which may be usefull across notebooks
# after looking for ways how to import functions from other notebooks I came accross
# https://github.com/grst/nbimporter and
# https://stackoverflow.com/questions/19564625/how-can-i-import-from-another-ipython-notebook
# I decided to use a seperate python file to collect my functions

import h5py
import holoviews as hv
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.stats import norm, binned_statistic_2d
import scipy.integrate as integrate


def hist2D_vmi(
    df: pd.DataFrame,
    p: dict,
    bins=(range(256), range(256)),
    sigma: int = None,
    weights: str = None,
) -> hv.Image:
    """plot a 2D histogram with the axis define as parameters
    default settings for the graph are log-scale for z-axis

    Parameters
    ----------
    df : pandas.DataFrame
        dataframe which contains the data
    p : dictionary
        dictionary containing the necessary parameters for the plot and data selection
        keys
        ----
        x : string
            name for the corresponding x-axes provided in `df`
        y : string
            name for the correspondig y-axes provided in `df`
        title : string
            the title of the plot
        xlabel : string
        ylabel : string
    bins : number, array
        bins for the `numpy.histogram2d`
    sigma : float, optional
        number for optional Gaussian smoothing of histogram
    weights : string, optional
        name of the corresponding column in `df` which should be used as weights for the histogram

    Return
    ------
    holoviews.Image

    See also
    --------
    holoviews.Image
    numpy.histogram2d
    """
    weights_data = df[weights] if weights is not None else None
    xy_hist, x_bins, y_bins = np.histogram2d(
        df[p["x"]], df[p["y"]], bins=bins, weights=weights_data
    )
    if sigma is not None:
        image = gaussian_filter(xy_hist.T[::-1], sigma=sigma)
    else:
        image = xy_hist.T[::-1]
    hist2d = hv.Image(
        image, bounds=(x_bins[0], y_bins[0], x_bins[-1], y_bins[-1])
    ).opts(
        axiswise=True,
        logz=True,
        clim=(0.1, None),
        title=p["title"],
        xlabel=p["xlabel"],
        ylabel=p["ylabel"],
    )

    return hist2d  # _slice_1st + hist2d_slice_all


def TOF_spectrum(data, title="DBSCAN", bins=15_000, alpha=1):
    """plot TOF spectrum"""
    x_hist, x_bins = np.histogram(data["tof"], bins=bins)
    tof = hv.Histogram((x_hist, x_bins), label=title).opts(
        xlabel="TOF (µs)", title=title
    )
    return tof.opts(alpha=alpha)


def get_data_pd(fname: str) -> (pd.DataFrame, pd.DataFrame):
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
            size = f['centroided/clustersize'][:]

        raw_data = pd.DataFrame(
            np.column_stack((rawNr, rawTof, rawTot, rawX, rawY)),
            columns=("nr", "tof", "tot", "x", "y"),
        )
        cent_data = pd.DataFrame(
            np.column_stack((centNr, centTof, centTot, centX, centY, size)),
            columns=("nr", "tof", "tot", "x", "y", 'size'),
        )
        return raw_data, cent_data
    except:
        print(f'key not known or file "{fname}" not existing')


def gauss_fwhm(x, *p):
    A, mu, fwhm = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * (fwhm ** 2) / (4 * 2 * np.log(2))))

def gauss_fwhm_double(x, *p):
    '''function to fit 2 Gauss peaks'''
    A, mu1, fwhm1, B, mu2, fwhm2 = p
    return A * np.exp(-((x - mu1) ** 2) / (2.0 * (fwhm1 ** 2) / (4 * 2 * np.log(2)))) + B * np.exp(-((x - mu2) ** 2) / (2.0 * (fwhm2 ** 2) / (4 * 2 * np.log(2))))

def gauss_fwhm_tripple(x, *p):
    '''function to fit 3 Gauss peaks'''
    A, mu1, fwhm1, B, mu2, fwhm2, C, mu3, fwhm3 = p
    return (A * np.exp(-((x - mu1) ** 2) / (2.0 * (fwhm1 ** 2) / (4 * 2 * np.log(2)))) + 
            B * np.exp(-((x - mu2) ** 2) / (2.0 * (fwhm2 ** 2) / (4 * 2 * np.log(2)))) +
            C * np.exp(-((x - mu3) ** 2) / (2.0 * (fwhm3 ** 2) / (4 * 2 * np.log(2)))))


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

def print_tpx_hdf5(fname):
    with h5py.File(fname, 'r') as f:
        print(f'groups: {f.keys()}')
        for key, val  in f.attrs.items():
            print(f'-> {key}: {val}')  
        print()
        for grp in f.keys():
            print(f'{grp} group: {f[grp].keys()}')
            for key, val  in f[grp].attrs.items():
                print(f'-> {key}: {val}')  
            for ds in f[grp].keys():
                for key, val  in f[grp][ds].attrs.items():
                    print(f'   {ds} -> {key}: {val}')
            for i in f[grp].items():
                print(f'   {i}')
        print('   |')
        print('   |')
        for grp in f['timing'].keys():
            grpStr = f'timing/{grp}'
            print(f'   -> {grpStr} group: {f[grpStr].keys()}')
            for key, val  in f[grpStr].attrs.items():
                print(f'      -> {key}: {val}')

            for ds in f[f'{grpStr}'].keys():
                for key, val  in f[grpStr][ds].attrs.items():
                    print(f'          {ds} -> {key}: {val}')
            for i in f[grpStr].items():
                print(f'          {i}')
                
def nano_to_datetime(nano):
    '''https://stackoverflow.com/questions/15649942/how-to-convert-epoch-time-with-nanoseconds-to-human-readable'''
    import datetime
    sec = nano*1e-9
    dt = datetime.datetime.fromtimestamp(sec)
    return dt.strftime('%Y-%m-%dT%H:%M:%S.%f')
    