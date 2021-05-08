from skimage.feature import blob_dog

from pipeline.centroiding_steps.Centroiding2D import Centroiding2D


class DifferenceOfGaussianCentroiding(Centroiding2D):

    def __init__(self, min_sigma=1, max_sigma=50, sigma_ratio=1.6, threshold=2.0, overlap=0.5,
                 exclude_border=False, *args, **kwargs):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma  # 3
        self.sigma_ratio = sigma_ratio
        self.threshold = threshold  # 100
        self.overlap = overlap
        self.exclude_border = exclude_border
        super(DifferenceOfGaussianCentroiding, self).__init__(*args, **kwargs)

    def find_centres(self, image_2D):
        blobs_dog = blob_dog(image_2D, min_sigma=self.min_sigma, max_sigma=self.max_sigma, sigma_ratio=self.sigma_ratio,
                             threshold=self.threshold, overlap=self.overlap, exclude_border=self.exclude_border)
        return blobs_dog[..., :2]
