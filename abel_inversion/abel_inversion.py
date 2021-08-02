import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from scipy import constants
import pandas as pd
import abel
import sys
plt.close('all')

vel_max = 15
nbins = 200
old = False
if old:
    data = np.load('/Users/strippel/CFEL/beamtimes/Desy_TPX-FlashSep20/data/N2_forAbelInversion.npz')
    hist = data['xy_hist']
else:
    df = pd.read_hdf('/Users/strippel/CFEL/beamtimes/Desy_TPX-FlashSep20/data/N2_forAbelInversion.h5')
    df1 = df.query(f"2.35 < tof < 2.55")
    bins = np.linspace(-vel_max, vel_max, nbins)
    hist, x_bins, y_bins = np.histogram2d(df1["v_x"] * 1e-3, df1["v_y"] * 1e-3, bins=bins)



print(hist.shape)
#data['x_bins']
#data['y_bins’]

plt.figure('Original', figsize=[6,6])
axis_lim = [x_bins[0], x_bins[-1], y_bins[0], y_bins[-1]]
ax_im0 = plt.imshow(hist.T,cmap="inferno", vmin=0, vmax=100, extent=axis_lim)
#ax_im0 = plt.imshow(hist.T,cmap="inferno", norm=LogNorm())
plt.xlim(-12,12)
plt.ylim(-12,12)
plt.grid(b=None)
plt.savefig('mq14.pdf')

plt.figure('Abel inverted image', figsize=[6,6])
inverse_abel = abel.Transform(hist, direction='inverse', method='rbasex').transform
#ax_im_inverse = plt.imshow(inverse_abel.T,cmap="inferno", norm=LogNorm(0.1,5), extent=[-12, 12, -12, 12])
ax_im_inverse = plt.imshow(inverse_abel.T,cmap="inferno", vmin=0, vmax=2, extent=axis_lim)
plt.grid(b=None)
plt.xlim(-12,12)
plt.ylim(-12,12)
plt.colorbar(ax_im_inverse)
plt.savefig('Abel-inverted.pdf')

fontsize=12
conversion=vel_max/nbins*2
radial = abel.tools.vmi.angular_integration(inverse_abel, Jacobian=False)
radial = np.array(radial)
#radial[1]/=radial[0]
plt.figure('Radial distribution', figsize=[12,4])
radial[0]*=conversion
print(radial[0].shape)
plt.plot(radial[0], radial[1])
plt.xlim(0,12)
plt.ylim(0,20)
plt.savefig('radial-distribution.pdf')


e = constants.value(u'elementary charge')
amc = constants.value(u'atomic mass constant')
plt.figure('Energy distribution', figsize=[12,4])
energy = 0.5*14*amc/e*(radial[0]*1000)**2
plt.plot(energy, radial[1])
plt.grid(b=None)
plt.xlim(0,10)
plt.ylim(0,20)
plt.savefig('energy-distribution.pdf')


plt.show()
