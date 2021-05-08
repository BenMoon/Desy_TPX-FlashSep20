from abc import abstractmethod
import logging

import numpy as np

from pipeline.PipelineStepGrouped import PipelineStepGrouped
from pipeline.ProcessingIn2D import ProcessingIn2D
from pipeline.gaussian_fitting.GaussianFitting import GaussianFitting
from pipeline.gaussian_fitting.GaussianFittingDummy import GaussianFittingDummy

logHandler = logging.StreamHandler()
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logHandler.setFormatter(logFormatter)
logger = logging.getLogger(__name__)
logger.addHandler(logHandler)
logger.setLevel(logging.ERROR)


class Centroiding2D(PipelineStepGrouped, ProcessingIn2D):

    def __init__(self, strategy_2D_mapping=ProcessingIn2D.build_2D_histogram_reduced,
                 strategy_gaussian_fitting: GaussianFitting = GaussianFittingDummy(),
                 *args, **kwargs):
        self.strategy_2D_mapping = strategy_2D_mapping
        self.strategy_gaussian_fitting = strategy_gaussian_fitting
        super(Centroiding2D, self).__init__(*args, **kwargs)

    @abstractmethod
    def find_centres(self, image_2D):
        pass

    def process_group(self, group):
        logger.debug(group)
        image_2D, shift_x, shift_y = self.strategy_2D_mapping(group)
        center_points = self.find_centres(image_2D)
        center_points = self.strategy_gaussian_fitting.fit(image_2D, center_points)

        if len(center_points) > 0:
            center_points[:, 0] = center_points[:, 0] + shift_x
            center_points[:, 1] = center_points[:, 1] + shift_y

            center_points_3D = Centroiding2D.map_2D_points_to_3D(group, center_points)

            return center_points_3D
        else:
            return np.empty((0, 0))
