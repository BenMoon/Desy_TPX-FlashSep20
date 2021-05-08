from functools import partial
from itertools import chain

import numpy as np

from pipeline.PipelineStep import PipelineStep
from pipeline.PipelineStepGrouped import PipelineStepGrouped


class PipelineStepGroupedMergeColumns(PipelineStepGrouped):
    def __init__(self, group_by_index_2=PipelineStep.LABEL_INDEX + 1, *args, **kwargs):
        self.group_by_index_2 = group_by_index_2
        super(PipelineStepGroupedMergeColumns, self).__init__(*args, **kwargs)

    def __combine_labels(self, clusters_grouped_1):
        partial_numpy_group_by = partial(
            PipelineStepGrouped.numpy_group_by, column_index=self.group_by_index_2
        )
        clusters_grouped_2 = chain.from_iterable(map(partial_numpy_group_by, clusters_grouped_1))

        label_counter = 0
        clusters_grouped = []
        for index, cluster in enumerate(clusters_grouped_2):
            label = label_counter
            if (cluster[..., self.group_by_index_2] == -1).any():
                label = -1
            else:
                label_counter += 1
            clusters_grouped.append(
                np.column_stack(
                    (cluster[..., : self.group_by_index], np.repeat(label, len(cluster)))
                )
            )

        return np.concatenate(clusters_grouped)

    def perform(self, data):
        data = super().perform(data)
        clusters_grouped = PipelineStepGrouped.numpy_group_by(data, self.group_by_index)
        voxels_labels_combined = self.__combine_labels(clusters_grouped)
        return voxels_labels_combined[voxels_labels_combined[..., PipelineStep.LABEL_INDEX] != -1]
