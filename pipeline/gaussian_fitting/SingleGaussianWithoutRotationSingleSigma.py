import numpy as np

from pipeline.gaussian_fitting.SingleGaussianFit import SingleGaussianFit


class SingleGaussianWithoutRotationSingleSigma(SingleGaussianFit):

    @staticmethod
    def __2D_gauss(data_tuple, x0, y0, sigma, A):
        """Function to fit, returns 2D gaussian function as 1D array
        2D Function from Wikipedia: https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function.
        But without different sigmas for x and y because the blobs should be squared in this case. In Addition the
        rotations were removed all gaussians will be circular and rotations will have no effect.
        """
        x, y = data_tuple

        a = 1 / (sigma ** 2)
        b = 0
        c = a

        a = a * ((x - x0) ** 2)
        b = 2 * b * (x - x0) * (y - y0)
        c = c * ((y - y0) ** 2)
        z = A * np.exp(-(a + b + c))
        return z.ravel()

    def get_2D_gaussian_function(self):
        return self.__2D_gauss

    def initial_guess(self, image, center):
        center_x, center_y = int(center[1]), int(center[0])
        initial_x0, initial_y0 = center_x, center_y
        initial_sigma = 1.0
        initial_A = image[center_y, center_x]
        return initial_x0, initial_y0, initial_sigma, initial_A

    def get_results(self, p_opt):
        result_x0, result_y0, result_sigma, result_A = p_opt
        return [result_y0, result_x0, result_sigma, result_A]
