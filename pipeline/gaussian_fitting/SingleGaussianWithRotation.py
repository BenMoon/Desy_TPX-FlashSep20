import numpy as np

from pipeline.gaussian_fitting.SingleGaussianFit import SingleGaussianFit


class SingleGaussianWithRotation(SingleGaussianFit):

    @staticmethod
    def __2D_gauss(data_tuple, A, x0, y0, sigma_x, sigma_y, theta):
        x, y = data_tuple
        x0 = float(x0)
        y0 = float(y0)
        a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
        b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
        c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
        g = A * np.exp(- (a * ((x - x0) ** 2) + 2 * b * (x - x0) * (y - y0) + c * ((y - y0) ** 2)))
        return g.ravel()

    def get_2D_gaussian_function(self):
        return self.__2D_gauss

    def initial_guess(self, image, center):
        center_x, center_y = int(center[1]), int(center[0])
        initial_A = image[center_y, center_x]
        initial_x0, initial_y0 = center_x, center_y
        initial_sigma_x, initial_sigma_y, initial_theta = 1.0, 1.0, 0
        return initial_A, initial_x0, initial_y0, initial_sigma_x, initial_sigma_y, initial_theta

    def get_results(self, p_opt):
        result_A, result_x0, result_y0, result_sigma_x, result_sigma_y, result_theta = p_opt
        return [result_x0, result_y0, (result_sigma_x + result_sigma_y) / 2, result_A]
