from skimage.feature import blob_doh

from pipeline.centroiding_steps.Centroiding2D import Centroiding2D


class DeterminantOfHessianCentroiding(Centroiding2D):

    def __init__(self, min_sigma=1, max_sigma=30, num_sigma=10, threshold=0.01, overlap=0.5, log_scale=False, *args, **kwargs):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma  ##2.2
        self.num_sigma = num_sigma
        self.threshold = threshold  ##100000
        self.overlap = overlap
        self.log_scale = log_scale
        super(DeterminantOfHessianCentroiding, self).__init__(*args, **kwargs)

    def find_centres(self, image_2D):
        blobs_doh = blob_doh(image_2D, min_sigma=self.min_sigma, max_sigma=self.max_sigma, num_sigma=self.num_sigma,
                             threshold=self.threshold, overlap=self.overlap, log_scale=self.log_scale)
        return blobs_doh[:, :2]
