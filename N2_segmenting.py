# necessary imports
import h5py
import numpy as np
from multiprocessing.pool import Pool
import multiprocessing as mp

from pipeline.Pipeline import Pipeline
from pipeline.clustering_steps.DBSCANClustering import DBSCANClustering
from pipeline.centroiding_steps.LaplacianOfGaussianCentroiding import (
    LaplacianOfGaussianCentroiding,
)
from pipeline.centroiding_steps.PeersCentroiding import PeersCentroiding
from pipeline.centroiding_steps.CenterOfMassCentroiding import CenterOfMassCentroiding
from pipeline.clustering_steps.PeersClustering import PeersClustering
from pipeline.clustering_steps.LaplacianOfGaussianClustering import LaplacianOfGaussianClustering
from pipeline.label_allocation.NeighborsInDistanceAllocationStrategy import (
    NeighborsInDistanceAllocationStrategy,
)


def evaluate_single_trigger_dbscan(trigger_nr: int) -> np.array:
    """fetch trigger event data and find clusters
    trigger_nr: integer to select the trigger number
    """
    # print(trigger_nr)
    with h5py.File(fname, "r") as f:
        # print(f.filename)
        trigger = f["raw/trigger nr"][:]
        mask = trigger == trigger_nr
        trigger = trigger[mask]
        x = f["raw/x"][mask]
        y = f["raw/y"][mask]
        tof = f["raw/tof"][mask] * 1e6
        tot = f["raw/tot"][mask]
    mask = tof < 40
    # raw_data = pd.DataFrame(data=np.column_stack((trigger[mask], x[mask], y[mask], tof[mask], tot[mask])), columns=['nr', 'x', 'y', 'tof', 'tot'])
    # print(f"cluster size: {datasets['clustersize'][trigger_nr]}”)

    X = np.column_stack((x, y, tof, tot))

    # start crunching numbers
    dbscan_clustering = DBSCANClustering(
        eps=1.5,
        min_samples=4,
        threshold=25,
        tof_scaling_factor=20 / 1000  # Tobis parameters
        # eps=2, min_samples=5, threshold=25, tof_scaling_factor=1e1  # 20 / 1000 # parameters from raw converter
    )
    log_clustering = LaplacianOfGaussianClustering(
        strategy_label_allocation=NeighborsInDistanceAllocationStrategy()
    )
    pipeline_dbscan_log = Pipeline(
        pipeline_steps=[dbscan_clustering, log_clustering, CenterOfMassCentroiding()]
    )

    try:
        # catch triggers where there are no cluters found in trigger frame
        # TODO: this should be changed, probably best to calculate several frames at the same time
        centroides_dbscan = pipeline_dbscan_log.run(X)

        trigger_nr_dbscan = np.repeat(trigger_nr, len(centroides_dbscan))
        return np.column_stack((trigger_nr_dbscan, centroides_dbscan))
    except:
        return None  # np.array(5 * [np.nan])


def evaluate_single_trigger_peer(trigger_nr: int) -> np.array:
    """fetch trigger event data and find clusters"""
    with h5py.File(fname, "r") as f:
        # print(f.filename)
        trigger = f["raw/trigger nr"][:]
        mask = trigger == trigger_nr
        trigger = trigger[mask]
        x = f["raw/x"][mask]
        y = f["raw/y"][mask]
        tof = f["raw/tof"][mask] * 1e6
        tot = f["raw/tot"][mask]
    mask = tof < 40
    # raw_data = pd.DataFrame(data=np.column_stack((trigger[mask], x[mask], y[mask], tof[mask], tot[mask])), columns=['nr', 'x', 'y', 'tof', 'tot'])
    # print(f"cluster size: {datasets['clustersize'][trigger_nr]}”)

    tw = np.load('out/timewalk_raw_N2.npy')
    tof -= tw[np.int_(tot // 25 -1)]*1e9
    X = np.column_stack((x, y, tof, tot))
    X = X[X[:, 2].argsort()]

    # start crunching numbers
    # Aufruf für Peers Algorithmus:
    peer_clustering = PeersClustering(max_dist_tof=0.11, min_cluster_size=3)
    pipeline_peers = Pipeline(pipeline_steps=[peer_clustering, CenterOfMassCentroiding()])
    centroids_peer = pipeline_peers.run(X)

    trigger_nr_peer = np.repeat(trigger_nr, len(centroids_peer))
    return np.column_stack((trigger_nr_peer, centroids_peer))


# load data
fname = "out/ion-run_0016_20200903-2202.hdf5"
with h5py.File(fname, "r") as f:
    triggers = f["raw/trigger nr"][:]
datasets = np.unique(triggers)[:]

cpu_count = mp.cpu_count()

# for idx in range(71786-2000,71786-1000):
# for idx in range(len(datasets)):
#    results = evaluate_single_trigger_dbscan(datasets[idx])
"""
# investigate problem where there are no cluters found in trigger frame
# problematic frames are 34443 71868
try:
    results = evaluate_single_trigger_dbscan(datasets[34443])
except:
    results = np.array(5 * [np.nan])
"""

######
# DBSCAN
# logger.debug(f'starting {cpu_count} threads')
"""
with Pool(cpu_count) as p:
    results = p.map(evaluate_single_trigger_dbscan, datasets[:])
# logger.debug('All Treads finished the calculation')
clusters_dbscan = np.concatenate([i for i in results if i is not None])

# save data
np.save("out/ion-run_0016_20200903-2202_LoG-rawConv.npy", clusters_dbscan)
"""

######
# Peer
# logger.debug(f'starting {cpu_count} threads')
with Pool(cpu_count) as p:
    results = p.map(evaluate_single_trigger_peer, datasets[:])
# logger.debug('All Treads finished the calculation')
clusters_peer = np.concatenate(results)

# save data
np.save("out/ion-run_0016_20200903-2202_peer.npy", clusters_peer)

