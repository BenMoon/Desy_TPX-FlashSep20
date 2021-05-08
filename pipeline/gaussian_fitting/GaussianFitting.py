from abc import ABC, abstractmethod


class GaussianFitting(ABC):

    @abstractmethod
    def fit(self, image_2D, centres):
        pass