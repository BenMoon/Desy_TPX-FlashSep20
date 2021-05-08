from skimage.feature import blob_log

from pipeline.clustering_steps.Clustering2D import Clustering2D


class LaplacianOfGaussianClustering(Clustering2D):

    def __init__(self, min_sigma=1, max_sigma=10, num_sigma=20, threshold=0.2, overlap=0.5, log_scale=False, exclude_border=False, *args, **kwargs):
        self.min_sigma = min_sigma
        self.max_sigma = max_sigma
        self.num_sigma = num_sigma
        self.threshold = threshold
        self.overlap = overlap
        self.log_scale = log_scale
        self.exclude_border = exclude_border
        super(LaplacianOfGaussianClustering, self).__init__(*args, **kwargs)

    def find_centres(self, image_2D):
        blobs_log = blob_log(image_2D, min_sigma=self.min_sigma, max_sigma=self.max_sigma, num_sigma=self.num_sigma,
                             threshold=self.threshold, overlap=self.overlap, log_scale=self.log_scale,
                             exclude_border=self.exclude_border)
        return blobs_log# [:, :2]