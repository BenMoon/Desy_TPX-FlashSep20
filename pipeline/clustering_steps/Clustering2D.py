from abc import abstractmethod

import numpy as np

from pipeline.PipelineStepGrouped import PipelineStepGrouped
from pipeline.PipelineStepGroupedMergeColumns import PipelineStepGroupedMergeColumns
from pipeline.ProcessingIn2D import ProcessingIn2D
from pipeline.label_allocation.AllocationStrategy import AllocationStrategy
from pipeline.label_allocation.NearestNeighborAllocationStrategy import NearestNeighborAllocationStrategy


class Clustering2D(PipelineStepGroupedMergeColumns, ProcessingIn2D):

    def __init__(self, size_limit=0, shape_ratio_limit=0, strategy_2D_mapping=ProcessingIn2D.build_2D_image_reduced,
                 strategy_label_allocation: AllocationStrategy = NearestNeighborAllocationStrategy(),
                 *args, **kwargs):
        self.size_limit = size_limit
        self.shape_ratio_limit = shape_ratio_limit
        self.strategy_2D_mapping = strategy_2D_mapping
        self.strategy_label_allocation = strategy_label_allocation
        super(Clustering2D, self).__init__(*args, **kwargs)

    @abstractmethod
    def find_centres(self, image_2D):
        pass

    def process_group(self, group):
        num_rows, _ = group.shape

        if num_rows > self.size_limit:
            image_2D, shift_x, shift_y = self.strategy_2D_mapping(group)
            center_points = self.find_centres(image_2D)

            if len(center_points) > 0:
                center_points[:, 0] = center_points[:, 0] + shift_x
                center_points[:, 1] = center_points[:, 1] + shift_y

                return self.strategy_label_allocation.allocate(group, center_points)
            else:
                return np.column_stack((group, np.repeat(-1.0, num_rows)))
        else:
            return np.column_stack((group, np.repeat(PipelineStepGrouped.INITIAL_LABEL, num_rows)))
