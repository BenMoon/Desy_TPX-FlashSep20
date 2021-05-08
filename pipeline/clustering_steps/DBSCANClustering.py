from sklearn.cluster import DBSCAN as sklearn_dbscan
import numpy as np

from pipeline.PipelineStep import PipelineStep
from pipeline.PipelineStepGroupedMergeColumns import PipelineStepGroupedMergeColumns


class DBSCANClustering(PipelineStepGroupedMergeColumns):

    def __init__(self, eps=2.85, min_samples=5, tof_scaling_factor=1, threshold=100, *args, **kwargs):
        self.eps = eps
        self.min_samples = min_samples
        self.tof_scaling_factor = tof_scaling_factor
        self.threshold = threshold
        super(DBSCANClustering, self).__init__(*args, **kwargs)

    def __scale_tof(self, tof):
        return tof * self.tof_scaling_factor

    def process_group(self, group):
        above_threshold = group[group[..., PipelineStep.TOT_INDEX] > self.threshold]
        below_threshold = group[group[..., PipelineStep.TOT_INDEX] <= self.threshold]
        X = above_threshold
        X = np.column_stack((X[..., PipelineStep.X_INDEX],
                             X[..., PipelineStep.Y_INDEX],
                             self.__scale_tof(X[..., PipelineStep.TOF_INDEX])))
        clustering = sklearn_dbscan(eps=self.eps, min_samples=self.min_samples).fit(X)
        voxels_with_labels = np.column_stack((above_threshold, clustering.labels_))
        outlier_labels = np.repeat(-1, below_threshold.shape[0])
        outliers_with_labels = np.column_stack((below_threshold, outlier_labels))
        return np.concatenate((voxels_with_labels, outliers_with_labels), axis=0)
