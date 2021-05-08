from abc import abstractmethod
from itertools import chain
import logging

import numpy as np

from pipeline.PipelineStep import PipelineStep

logHandler = logging.StreamHandler()
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s"
)
logHandler.setFormatter(logFormatter)
logger = logging.getLogger(__name__)
logger.addHandler(logHandler)
logger.setLevel(logging.ERROR)


class PipelineStepGrouped(PipelineStep):

    INITIAL_LABEL = 0

    def __init__(self, group_by_index=PipelineStep.LABEL_INDEX, *args, **kwargs):
        self.group_by_index = group_by_index
        super(PipelineStepGrouped, self).__init__(*args, **kwargs)

    @staticmethod
    def numpy_group_by(array, column_index):
        array = array[
            array[:, column_index].argsort()
        ]  # Sorting by the column first for the grouping to work out
        # must be sorted by the column to do this!
        return np.split(array, np.unique(array[:, column_index], return_index=True)[1][1:])

    def perform(self, data):
        num_rows, num_cols = data.shape
        if self.group_by_index >= num_cols:
            # initialize with a single label (all observations belong to the same group)
            group_column = np.repeat(PipelineStepGrouped.INITIAL_LABEL, num_rows)
            enriched_data = np.column_stack((data, group_column))
        else:
            enriched_data = data

        data_grouped = PipelineStepGrouped.numpy_group_by(enriched_data, self.group_by_index)
        logger.debug(data_grouped)
        return np.array(list(chain.from_iterable(map(self.process_group, data_grouped))))

    @abstractmethod
    def process_group(self, group):
        pass
