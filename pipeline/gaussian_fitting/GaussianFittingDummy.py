from pipeline.gaussian_fitting.GaussianFitting import GaussianFitting


class GaussianFittingDummy(GaussianFitting):

    def fit(self, image_2D, centres):
        return centres
