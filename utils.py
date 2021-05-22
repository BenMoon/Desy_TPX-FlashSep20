# this file contains a collections of functions which may be usefull across notebooks
# after looking for ways how to import functions from other notebooks I came accross
# https://github.com/grst/nbimporter and 
# https://stackoverflow.com/questions/19564625/how-can-i-import-from-another-ipython-notebook
# I decided to use a seperate python file to collect my functions

import pandas as pd
import numpy as np
from scipy.ndimage import gaussian_filter
import holoviews as hv


def hist2D_vmi(
    df: pd.DataFrame,
    p: dict,
    bins=(range(256), range(256)),
    sigma: int = None,
    weights: str = None,
) -> hv.Image:
    '''plot a 2D histogram with the axis define as parameters
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
    '''
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
    x_hist, x_bins = np.histogram(data["tof"], bins=bins)
    tof = hv.Histogram((x_hist, x_bins), label=title).opts(
        xlabel="TOF (µs)", title=title
    )
    return tof.opts(alpha=alpha)