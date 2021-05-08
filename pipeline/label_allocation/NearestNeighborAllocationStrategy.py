from abc import ABC
from functools import partial

from scipy.spatial import KDTree
import numpy as np

from pipeline.PipelineStep import PipelineStep
from pipeline.ProcessingIn2D import ProcessingIn2D


class NearestNeighborAllocationStrategy(ABC):

    def allocate(self, voxels, centres):
        center_points = centres[..., :PipelineStep.Y_INDEX + 1]
        center_points_3D = ProcessingIn2D.map_2D_points_to_3D(voxels, center_points[..., :2])
        kd_tree = KDTree(center_points_3D[..., :PipelineStep.TOF_INDEX + 1])
        labels = np.apply_along_axis(
            partial(NearestNeighborAllocationStrategy.__get_label_nearest_neighbor, kd_tree=kd_tree), axis=1, arr=voxels)
        return np.column_stack((voxels, labels))

    @staticmethod
    def __get_label_nearest_neighbor(point, kd_tree):
        nearest_neighbor_index = NearestNeighborAllocationStrategy.__query_nearest_neighbor_index(point, kd_tree)
        return nearest_neighbor_index

    @staticmethod
    def __query_nearest_neighbor_index(point, kd_tree):
        x, y, tof = point[PipelineStep.X_INDEX], point[PipelineStep.Y_INDEX], point[
            PipelineStep.TOF_INDEX]
        dist, nearest_neighbor_index = kd_tree.query([x, y, tof])
        return nearest_neighbor_index