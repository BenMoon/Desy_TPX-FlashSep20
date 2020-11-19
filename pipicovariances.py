
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

# =============================================================================
# global parameters
# =============================================================================

cmap = 'Spectral_r' #color map for velocity maps

# =============================================================================
# read in data
# =============================================================================

import h5py

#load data:
file = 'pyrrole_3_EL' #'pyrole_25_EL' ##'pyrrole_4_EL' pyrrole_4_EL
#file = 'pyrrolenew__42_69_EL'#'pyrrolenew__42_51_EL'#'pyrrolenew__42_69_EL'#'pyrrole__44_45_EL'#'pyrrolenew__42_69_EL'#'pyrrole__44_45_EL'#'pyrrolenew__42_69_EL'#'pyrrole__100_109_EL'#'pyrrole__100_109_EL'#'pyrrole__42_51_EL'#'pyrrole__44_45_EL'#'pyrrole__17_18_19_20_EL'#'pyrrole__100_101_bg'pyrrole__44_45_EL
with h5py.File(file + '.h5', 'r') as h5:
    hits = h5['hits'][:]
    
# =============================================================================
# select foreground shots according to LED flag
# =============================================================================
    
from numpy import sqrt
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

vmax_led = 6

#create raw-velocity histogram of periphery counts:
v0_per = np.array([128, 128])[None, None, :]
vr = sqrt(np.sum((hits[..., 1:] - v0_per)**2, -1))
b_g0 = hits[..., 0] > 0
b_per = vr > 128
#bins_per = np.linspace(-25, 281, 307)
#h_per = np.histogram2d(hits[b_per*b_g0, 1], hits[b_per*b_g0, 2],
#                       bins=2*(bins_per,))[0]
#
##find LED position in velocity space:
#j_led = np.where(h_per == np.max(h_per))
#v_h = (bins_per[:-1] + bins_per[1:])/2
#v0_led = v_h[j_led[0][0]], v_h[j_led[1][0]]
#
##plot periphery velocity map:
#ext_per = 2*[bins_per[0], bins_per[-1]]
#plt.figure(figsize=(6.5, 5))
#plt.imshow(h_per.T, extent=ext_per, origin='lower', interpolation='nearest',
#           cmap=cmap, norm=mplc.LogNorm())
#c_led = plt.Circle(v0_led, vmax_led, color='green', lw=0.5, fill=False)
#plt.gca().add_artist(c_led)
#plt.colorbar(aspect=50)
#
##identify shots with LED flag:
#v_led = sqrt(np.sum((hits[..., 1:] - np.array(v0_led)[None, None, :])**2, -1))
#b_fg = np.any(v_led < vmax_led, 1)
#
##start boolean for ions to be selected:
#b_sel = b_g0 * b_fg[:, None] * (True ^ b_per)

#start boolean for ions to be selected:
b_sel = b_g0 * (True ^ b_per)

# =============================================================================
# time-of-flight-independent velocity-origin correction
# =============================================================================

coeff_v0 = (11.489, 113.454, 122.399, 129.16)
v00x, v00y = coeff_v0[-2:]
v0 = np.array([v00x, v00y])[None, None, :]

hits[..., 1:] -= v0

# =============================================================================
# filter out ions according to time-of-flight-indpendent velocity
# =============================================================================

#filter out chamber background according to raw 2D velocity:
vrmin = 10#10 #minimum radial raw velocity
b_sel[b_sel] = sqrt(np.sum(hits[b_sel, 1:]**2, -1)) > vrmin

#create raw-velocity histograms:
vrmax = 180
bins_vr = np.linspace(-vrmax, vrmax, 181)
h_vr = np.histogram2d(hits[b_sel, 1], hits[b_sel, 2], bins=(bins_vr, bins_vr))[0]

#plot velocity map:
ext_vr = 2*[bins_vr[0], bins_vr[-1]]
plt.figure(figsize=(6.5, 5))
plt.imshow(h_vr.T, extent=ext_vr, origin='lower', interpolation='nearest',
           cmap=cmap, norm=mplc.LogNorm())
c_vrmin = plt.Circle((0, 0), vrmin, color='green', lw=0.5, fill=False)
plt.gca().add_artist(c_vrmin)
plt.colorbar(aspect=50)
    
# %%=============================================================================
# time-of-flight to mass/charge calibration
# =============================================================================

#convert time-of-flight axis to µs:
hits[..., 0] *= 1e6

#define peaks for calibration:
peak_a              = 1.5238 #µs
peak_b              = 2.7935 #µs
mass_a              = 18 #u/e
mass_b              = 67 #u/e
#peak_a              = 1.51026 #µs
#peak_b              = 3.11096 #µs
#mass_a              = 18 #u/e
#mass_b              = 85 #u/e

t0 = (peak_b*(mass_a/mass_b)**0.5 - peak_a) / ((mass_a/mass_b)**0.5 - 1)
C  = mass_a / (peak_a - t0)**2

def tof2mz(array):
    return C * (array - t0)**2
    
def mz2tof(array):
    return np.sqrt(array/C) + t0

#create total time-of-flight spectrum:
Nbins_tof = 5000
s = hits.shape
hits_rav = hits[b_sel]
bins_tof = np.linspace(0, hits_rav[:, 0].max(), Nbins_tof+1)
tof_sum, _tof = np.histogram(hits_rav[:, 0], bins=bins_tof)
tof_sum[0] = 0 #eliminate empty fragment order slots from histogram
tof_sum = tof_sum/s[0] #normalise to hits/shot
t_sum = (_tof[:-1] + _tof[1:])/2

mz_sum = tof2mz(t_sum)

#plot total time-of-flight spectrum:
plt.figure(figsize=(8, 4))
plt.plot(t_sum, tof_sum, lw=1)
plt.xlabel(r'time of flight in $\mu$s')
plt.grid(ls=':')
plt.xlim([0.3, 3.2])
#plt.ylim([-1e-4, 1.6e-2])

#plot total mass/charge spectrum:
plt.figure(figsize=(8, 4))
plt.plot(mz_sum, tof_sum, lw=1)
plt.xlabel(r'mass/charge in u/e')
plt.ylabel('hits per shot')
plt.grid(ls=':')
plt.xlim([0, 90])
#plt.ylim([-1e-4, 2.6e-3]) pyrrole
plt.ylim([-1e-4, 2.0e-2]) #pyrrole-water
#%%============================================================================
# build pairs of coinciding ions
# =============================================================================

from itertools import combinations


def progress(j, Sj, n):
    'returns string for progress(j) of Sj-sliced computation at n points'
    th_pr = np.linspace(1/n, 1, n)*(Sj - 1 % 10)
    th_pr = th_pr.astype(int)
    if j in th_pr:
        i = np.where(th_pr == j)[0][0]
        per = int(round((i + 1)*100/n))
        out1 = str(per).rjust(3) + '% '
        out2 = '(' + str(j + 1) + '/' + str(Sj) + ')'
        print(out1 + out2)
    

#compute mass/charge ratios:
mz = tof2mz(hits[..., 0]) * b_sel

#mz = mz2tof(mz)#-t0

##identify shots for correlation analysis based on the number of fragments:
#N_frag = np.sum(b_sel, 1)
#b_shot = (N_frag > 0)# * (N_frag < 5)
#j_shot = np.where(b_shot)[0]

##allocate index array of coinciding ions:
#j_coin = np.array([], dtype=int).reshape(0, 3)

#allocate 2D expectation value map:
bins_mz = np.linspace(0, 80, 701) #73 #501
#bins_mz = np.linspace(0, 3.15, 1001) #73 #501
h_2b = np.zeros(2*(len(bins_mz)-1,))
N = len(mz)

#loop over shots:
for j in range(N):
    
#    #deduce ion-order indices:
#    j_mz = np.where(b_sel[j])[0]
    
    #create shot-wise mass/charge histogram:
    h_1d_j = np.histogram(mz[j], bins=bins_mz)[0]
    
    #add broadcasted product to 2D expectation value map:
    h_2b += h_1d_j[:, None] * h_1d_j[None, :]
    
#    #build 2-body ion pair combinations:
#    comb = np.array(list(combinations(j_mz, 2)))  
    
#    #infer minimum masses for all combination pairs (implying a charge of +1):
#    m_j = mz[j, comb]
#    
#    #compute sum of pair-wise minimum masses:
#    Sm_j = np.sum(m_j, -1)
#    
#    #identify pairs, whose minimum total mass surpass the parent mass:
#    b_Sm = Sm_j < M + 4
    
#    #add ion pairs to PIPICo map:
#    h_2b_j = np.histogram2d(mz[j, comb[:, 0]], mz[j, comb[:, 1]],
#                            bins=2*(bins_mz,))[0]
#    h_2b += h_2b_j**2
    
    #print progress:
    progress(j, N, 20)
    
#    #deduce 2D velocities for all combination pairs:
#    v_j = hits[j, comb, 1:]
#    
#    #infer masses for all combination pairs (implying a charge of +1):
#    m_j = mz[j, comb][..., None] * np.array(charge)[None, None, :]
#    
#    #compute sum momenta for all combination pairs:
#    p_sum = np.sum(m_j[..., None]*v_j[..., None, :], 1)
#    
#    #fill sum momenta into histogram:
#    h_Sp += np.histogram2d(p_sum[..., 0].ravel(), p_sum[..., 1].ravel(),
#                           bins=2*(bins_Sp,))[0]
#    
#    #identify pairs that fulfill momentum conservation:
#    p_rad = sqrt(np.sum((p_sum-p0_mc)**2, -1))
#    b_mc = p_rad < pmin_mc
#    
#    #get disjunction over charge axis:
#    b_mc = np.any(b_mc, 1)
#    
#    #deduce indices of ions that are part of a correlated pair:
#    j_mc = comb[b_mc]
    
#    #append indices of coinciding ions to array:
#    j_coin_j = np.concatenate((np.repeat(j, len(comb))[:, None], comb), -1)
#    j_coin = np.concatenate((j_coin, j_coin_j), 0)
    
#create 1D mass/charge spectrum:
h_1d = np.histogram(mz.ravel(), bins=bins_mz)[0]/N

#build uncorrelated 2D spectrum:
h_2d = h_1d[:, None] * h_1d[None, :]

#erase lower-right diagonal of uncorrelated 2D spectrum:
j1d = np.arange(len(bins_mz)-1)
jx, jy = np.meshgrid(j1d, j1d, indexing='ij')
h_2d[jx >= jy] = 0
h_2b[jx >= jy] = 0

#subtract correlated from uncorrelated map:
h_cov = h_2b/N - 1*h_2d #5

#%%============================================================================
# plot PIPICo map
# =============================================================================

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.ticker import MultipleLocator

##get ravelled mass/charge entries:
#mz1 = mz[(j_coin[:, 0], j_coin[:, 1])]
#mz2 = mz[(j_coin[:, 0], j_coin[:, 2])]
#mz_2b = np.append(mz1, mz2)

##create PIPICo map:
#h_2b = np.histogram2d(mz1, mz2, bins=(bins_mz, bins_mz))[0]

#plot PIPICo map:
ext_2b = 2*[bins_mz[0], bins_mz[-1]]
plt.figure(figsize=(6.5, 5))
ax = plt.subplot2grid((1, 1), (0, 0))
plt.imshow(h_cov.T, origin='lower', extent=ext_2b, interpolation='nearest',
           norm=mplc.LogNorm(), vmin=1e-5, vmax=1e-2,
#           norm=mplc.SymLogNorm(linthresh=1e-3), vmin=-10, vmax=10,
           cmap='afmhot_r')
plt.colorbar(aspect=40)
#plt.xlabel(r'mass/charge of 1st ion in u/e')
#plt.ylabel(r'mass/charge of 2nd ion in u/e')
plt.gca().xaxis.set_minor_locator(MultipleLocator(2))
plt.gca().yaxis.set_minor_locator(MultipleLocator(2))
#plt.grid()


#plt.hist2d(pip_x,pip_y,bins=bins,norm=mplc.LogNorm(),range=[[startx,startx+widthx],[starty,starty+widthy]],cmap='hot')

#region = [ 0,138,0, 138]  
#pyrrolewa 2819[ 126, 136,155, 225]  0
#pyrrolewa 39119 [ 126, 136,235,305]  
#pyrrole 3928 [ 155, 225,235,305]  1554  [90,110,320,390] 
#axi1 = inset_axes(ax, width='100%', height='100%',
#                  bbox_to_anchor=(0.61, 0.15, 0.38, 0.38),
#                  bbox_transform=ax.transAxes)
#axi1.imshow(h_cov[region[0]:region[1],region[2]:region[3]].T, origin='lower', 
#            extent=[bins_mz[region[0]],bins_mz[region[1]],bins_mz[region[2]],bins_mz[region[3]]], interpolation='nearest',
#           cmap='afmhot_r',vmin=1e-5, vmax=5e-3, norm=mplc.LogNorm())
#plt.xlim([23, 33])
#plt.ylim([33.5, 43.5])
#plt.gca().xaxis.set_major_locator(MultipleLocator(2))
#plt.gca().yaxis.set_major_locator(MultipleLocator(2))
#plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
#plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

#print(np.sum(h_cov[region[0]:region[1],region[2]:region[3]].T))
#
#axi2 = inset_axes(ax, width='100%', height='100%',
#                  bbox_to_anchor=(0.5, 0.105, 0.46, 0.23),
#                  bbox_transform=ax.transAxes)
#axi2.imshow(h_2b.T, origin='lower', extent=ext_2b, interpolation='nearest',
#           cmap='binary', norm=mplc.LogNorm())
#plt.xlim([17, 30])
#plt.ylim([36, 42.5])
#plt.gca().xaxis.set_major_locator(MultipleLocator(2))
#plt.gca().yaxis.set_major_locator(MultipleLocator(2))
#plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
#plt.gca().yaxis.set_minor_locator(MultipleLocator(1))

#axi1 = inset_axes(ax, width='100%', height='100%',
#                  bbox_to_anchor=(0.4, 0.1, 0.3, 0.3),
#                  bbox_transform=ax.transAxes)
#axi1.imshow(h_2b.T, origin='lower', extent=ext_2b, interpolation='nearest',
#           cmap='binary', norm=mplc.LogNorm())
#plt.xlim([23.5, 31.5])
#plt.ylim([35, 43])
#plt.gca().xaxis.set_major_locator(MultipleLocator(2))
#plt.gca().yaxis.set_major_locator(MultipleLocator(2))
#plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
#plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
#
#axi2 = inset_axes(ax, width='100%', height='100%',
#                  bbox_to_anchor=(0.73, 0.43, 0.25, 0.25),
#                  bbox_transform=ax.transAxes)
#axi2.imshow(h_2b.T, origin='lower', extent=ext_2b, interpolation='nearest',
#           cmap='binary', norm=mplc.LogNorm())
#plt.xlim([64.5, 68.5])
#plt.ylim([66.5, 70.5])
#plt.gca().xaxis.set_major_locator(MultipleLocator(2))
#plt.gca().yaxis.set_major_locator(MultipleLocator(2))
#plt.gca().xaxis.set_minor_locator(MultipleLocator(1))
#plt.gca().yaxis.set_minor_locator(MultipleLocator(1))
#plt.savefig('covmap_pyrrolew_4_el.pdf', bbox_inches='tight', dpi=1200)
#plt.savefig('covmap_pyrrolew_4261_100_el_1.pdf', bbox_inches='tight', dpi=1200)