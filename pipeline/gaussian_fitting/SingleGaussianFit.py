import logging
from abc import abstractmethod
from functools import partial

import numpy as np
import scipy.optimize as opt

from pipeline.gaussian_fitting.GaussianFitting import GaussianFitting

logHandler = logging.StreamHandler()
logFormatter = logging.Formatter("%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
logHandler.setFormatter(logFormatter)
logger = logging.getLogger(__name__)
logger.addHandler(logHandler)
logger.setLevel(logging.ERROR)


class SingleGaussianFit(GaussianFitting):

    def __init__(self, region_radius=2, *args, **kwargs):
        self.region_radius = region_radius
        super(SingleGaussianFit, self).__init__(*args, **kwargs)

    @abstractmethod
    def get_2D_gaussian_function(self):
        pass

    @abstractmethod
    def initial_guess(self, image, center):
        pass

    @abstractmethod
    def get_results(self, p_opt):
        pass

    def get_bounds(self, _image, _center):
        return (0, np.inf)

    def __fit_for_centre(self, image, center):
        """Get x, y, radius of a blob by 2D gaussian fitting
        Parameter:
            image_2D - image as numpy array
            center - initial x, y for the position of the gaussian
        Returns:
            x, y and radius for of the gaussian fit
        """
        height, width = image.shape

        x = range(0, width)
        y = range(0, height)
        x, y = np.meshgrid(x, y)

        y_data = image.ravel()

        try:
            p_opt, _p_cov = opt.curve_fit(self.get_2D_gaussian_function(), (x, y), y_data,
                                          p0=self.initial_guess(image, center),
                                          bounds=self.get_bounds(image, center))
        except RuntimeError as err:
            logger.debug(f'center {center} in image not found with gaussian fitting')
            logger.debug(err)
            return None

        return self.get_results(p_opt)

    def fit(self, image_2D, centres):
        tmp_fit_for_centre = partial(self.__fit_for_centre, image_2D)
        return np.array(list(filter(None, map(tmp_fit_for_centre, centres))))
