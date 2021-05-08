from math import sqrt

from pipeline.label_allocation.AllocationStrategy import AllocationStrategy

from scipy.spatial import KDTree
import numpy as np

from pipeline.PipelineStep import PipelineStep
from pipeline.clustering_steps.Clustering2D import Clustering2D


class NeighborsInDistanceAllocationStrategy(AllocationStrategy):

    def __init__(self, size_factor=sqrt(2), *args, **kwargs):
        self.size_factor = size_factor
        super(NeighborsInDistanceAllocationStrategy, self).__init__(*args, **kwargs)

    def allocate(self, voxels, centres):
        kd_tree = KDTree(voxels[..., :PipelineStep.TOF_INDEX + 1])
        labeled_points = np.column_stack((voxels, np.repeat(-1, voxels.shape[0])))
        label_number = 0
        for center in centres:
            radius = center[2]
            center_point_3D = Clustering2D.map_2D_point_to_3D(voxels, center[:2])
            for voxels_in_distance in kd_tree.query_ball_point(center_point_3D[..., :PipelineStep.TOF_INDEX + 1],
                                                               radius * self.size_factor):
                labeled_points[voxels_in_distance, PipelineStep.LABEL_INDEX + 1] = label_number
                label_number += 1
            # TODO: It should be possible for a single point to get multiple lables if it is in the range of multile
            #  centres. In this case the corresponding voxel should be doubled to get both labels
        return labeled_points
