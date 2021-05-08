from skimage import exposure
from skimage.morphology import extrema
import numpy as np

from pipeline.centroiding_steps.Centroiding2D import Centroiding2D


class ExtremaLocalH_MaximaCentroiding(Centroiding2D):

    @staticmethod
    def extract_points(d_arr):
        points = []
        for x, arr in enumerate(d_arr):
            for y, val in enumerate(arr):
                if val:
                    points.append([x, y])
        return points

    def __init__(self, h=0.05, *args, **kwargs):
        self.h = h
        super(ExtremaLocalH_MaximaCentroiding, self).__init__(*args, **kwargs)

    def find_centres(self, image_2D):
        image_2D = exposure.rescale_intensity(image_2D)

        h_maxima = extrema.h_maxima(image_2D, self.h)
        points = ExtremaLocalH_MaximaCentroiding.extract_points(h_maxima)

        return np.array(points)