import numpy as np
import h5py
import pandas as pd
from tqdm import tqdm

from sklearn.cluster import DBSCAN
#from joblib import Parallel, delayed

import holoviews as hv
from holoviews import opts
hv.extension('bokeh')

%pylab inline
#from matplotlib.colors import LogNorm
%config InlineBackend.figure_format ='retina'

rcParams['figure.figsize'] = (13.0, 6.)