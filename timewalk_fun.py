from scipy.optimize import curve_fit
from scipy.stats import norm

def getData(fname):
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
    return rawNr, rawTof, rawTot, rawX, rawY, centNr, centTof, centTot, centY, centX

def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-(x - mu) ** 2 / (2. * sigma ** 2))

def plot_TofTot(tof, tot, region, fname, **kwargs):
    # Filter for the calibration region we are looking at
    region_filter = (tof >= region[0]) & (tof <= region[1])
    tof_region = tof[region_filter]
    tot_region = tot[region_filter]

    # Find maximum tot
    max_tot_index = np.argmax(tot_region)

    # This is our 'correct' TOF
    center_tof = tof_region[max_tot_index]
    # Compute the time difference
    time_diff = tof_region - center_tof

    # Sample on a 2d histogram
    time_hist, tot_bins, time_bins, _ = hist2d(tot_region, time_diff, bins=(np.arange(tot_region.min(), tot_region.max() + 25, 25), 100), cmap='jet', **kwargs)
    title(f'{fname}')
    xlabel('TOT (ns)')
    ylabel('Time difference from center (ns)');
    
def plot_TofTot_hv(tof, tot, region, fname, **kwargs):
    # Filter for the calibration region we are looking at
    region_filter = (tof >= region[0]) & (tof <= region[1])
    tof_region = tof[region_filter]
    tot_region = tot[region_filter]

    # Find maximum tot
    max_tot_index = np.argmax(tot_region)

    # This is our 'correct' TOF
    center_tof = tof_region[max_tot_index]
    # Compute the time difference
    time_diff = tof_region - center_tof

    # Sample on a 2d histogram
    time_hist, tot_bins, time_bins = np.histogram2d(tot_region, time_diff, bins=(np.arange(tot_region.min(), tot_region.max() + 25, 25), 100))
    bin_edges = time_bins
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centres = bin_edges[:-1]

    #plt.imshow(time_hist.T, origin='lower', cmap='jet')
    return hv.Image(time_hist.T[::-1], bounds=(tot_bins[0], time_bins[0], tot_bins[-1], time_bins[-1])).opts(
        width=800, cmap='jet', title=fname, ylabel='diff TOF', xlabel='TOT', logz=True)

def compute_timewalk(tof, tot, region, maxTot_slice):
    tot_points = []
    time_walk_points = []

    # Filter for the calibration region we are looking at
    region_filter = (tof >= region[0]) & (tof <= region[1])
    tof_region = tof[region_filter]
    tot_region = tot[region_filter]

    # Find maximum tot
    max_tot_index = np.argmax(tot_region)

    # This is our 'correct' TOF
    center_tof = tof_region[max_tot_index]
    # Compute the time difference
    time_diff = tof_region - center_tof

    # Sample on a 2d histogram
    time_hist, tot_bins, time_bins = np.histogram2d(tot_region, time_diff, bins=(np.arange(tot_region.min(), tot_region.max() + 25, 25), 100))
    bin_edges = time_bins
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    bin_centres = bin_edges[:-1]
    
    total_bins = time_hist.shape[0]

    # For each bin
    for b in range(0, maxTot_slice):
        current_tot = time_hist[b]

        # Fit sampled tot region with gaussian
        A_guess = np.max(current_tot)
        center_guess = np.sum(current_tot * bin_centres) / np.sum(current_tot)
        sigma_guess = np.sqrt(np.sum(current_tot * np.square(bin_centres - center_guess)) / (np.sum(current_tot) - 1))
        p0 = [np.max(current_tot), center_guess, sigma_guess]
        try:
            coeff, var_matrix = curve_fit(gauss, bin_centres, current_tot, p0=p0)
        except:
            print(f"Counldn't do it in slice {b}")
            continue
        if np.isnan(coeff[2]):
            print(f"sigma is nan in slice {b}")
            coeff[1] = 0
            #continue
        
        time_walk_points.append(coeff[1])
        tot_points.append(tot_bins[b])

    image = hv.Image(time_hist.T[::-1], bounds=(tot_bins[0], time_bins[0], tot_bins[-1], time_bins[-1])).opts(
        width=1200, height=600, cmap='jet', ylabel='diff TOF', xlabel='TOT', logz=True, tools=['hover'])
    return np.array(tot_points), np.array(time_walk_points), image

def compute_tw_lookup(tof, tot, region, maxTot_slice=167, minTot=0, maxToT_1=4000, polyorder=12):
    #maxTot_slice = 190
    maxToT_1_idx = maxToT_1 // 25
    minTot_idx = minTot // 25
    tot_points, timewalk_points, hist = compute_timewalk(tof, tot, region, maxTot_slice)
    # first part of curve
    p1 = np.polyfit(tot_points[minTot_idx:]+12.5, timewalk_points[minTot_idx:], polyorder)
    x1 = np.arange(minTot, maxToT_1, 1)
    tw_lookup1 = np.polyval(p1, x1)
    # second part of curve
    p2 = np.polyfit(tot_points[maxToT_1_idx:]+12.5, timewalk_points[maxToT_1_idx:], 1)
    x2 = np.arange(maxToT_1, tot.max(), 1)
    tw_lookup2 = np.polyval(p2, x2)
    
    tot_lookup_table = np.zeros(0x3FF, dtype=np.float32)
    for x in range(0x3FF):
        try:
            if x <= (maxToT_1_idx):
                val = np.polyval(p1, ((x + 1) * 25))
                tot_lookup_table[x] = val
            else:
                val = np.polyval(p2, ((x + 1) * 25))
                tot_lookup_table[x] = val
        except:
            pass    

    a = hv.Curve((tot_points+12.5, timewalk_points)).opts(color='blue')
    b = hv.Curve((x1, tw_lookup1)).opts(color='orange', line_width=1)
    c = hv.Curve((x2, tw_lookup2)).opts(color='green', line_width=1)
    return hist * a * b * c, tot_lookup_table * 1E-9

def doCentroiding(shot, x, y, tof, tot, _epsilon=2, _samples=5):    
    tof_eps = 81920 * (25. / 4096)# * 1E-9

    tof_scale = _epsilon / tof_eps
    X = np.vstack((shot * _epsilon * 1000, x, y, tof * tof_scale)).transpose()
    dist = DBSCAN(eps=_epsilon, min_samples=_samples, metric='euclidean', n_jobs=15).fit(X)
    labels = dist.labels_ + 1

    label_filter = labels != 0 # ignore noise

    def cluster_properties(shot, x, y, tof, tot, labels):
        import scipy.ndimage as nd

        label_index = np.unique(labels)
        tot_max = np.array(nd.maximum_position(tot, labels=labels, index=label_index)).flatten()
        # tof_min = np.array(nd.minimum_position(tof,labels=labels,index=label_index)).flatten()
        # print(tot_max)
        tot_sum = nd.sum(tot, labels=labels, index=label_index)
        cluster_x = np.array(nd.sum(x * tot, labels=labels, index=label_index) / tot_sum).flatten()
        cluster_y = np.array(nd.sum(y * tot, labels=labels, index=label_index) / tot_sum).flatten()
        cluster_tof = np.array(nd.sum(tof * tot, labels=labels, index=label_index) / tot_sum).flatten() # timewalk
        cluster_tot = tot[tot_max]
        #cluster_tof = tof[tot_max] # no timewalk
        cluster_shot = shot[tot_max]

        return cluster_shot, cluster_x, cluster_y, cluster_tof, cluster_tot

    props = cluster_properties(shot[label_filter], x[label_filter], y[label_filter], tof[label_filter],
                                                tot[label_filter], labels[label_filter])

    XCentNew2  = pd.DataFrame(np.column_stack((props[0], props[1], props[2], props[3], props[4])), 
                 columns=['nr', 'x', 'y', 'tof', 'tot'])
    return XCentNew2