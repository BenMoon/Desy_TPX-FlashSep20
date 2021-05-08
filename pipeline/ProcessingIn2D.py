import itertools

import numpy as np

from pipeline.IndexPositions import IndexPositions


class ProcessingIn2D(IndexPositions):
    
    @staticmethod
    def build_2D_image_complete(X):
        """ Builds the complete 2D image for this cluster. Empty pixels on the edges are included. """
        image_2D = np.zeros((256, 256))
        image_2D[np.int_(X[..., IndexPositions.X_INDEX]), np.int_(X[..., IndexPositions.Y_INDEX])] += X[
            ..., IndexPositions.TOT_INDEX]

        shift_x = 0
        shift_y = 0

        return image_2D, shift_x, shift_y

    @staticmethod
    def build_2D_image_reduced(X):
        """ Builds a reduced 2D image for this cluster. Empty pixels will be excluded to possibly reduce the runtime.
            A shift for x and y will be returned that has to be used if the values will be used in the context of the complete image. """
        x_range = int(X[..., IndexPositions.X_INDEX].max() - X[..., IndexPositions.X_INDEX].min() + 1)
        y_range = int(X[..., IndexPositions.Y_INDEX].max() - X[..., IndexPositions.Y_INDEX].min() + 1)
        image_2D = np.zeros((x_range, y_range))
        image_2D[np.int_(X[..., IndexPositions.X_INDEX] - X[..., IndexPositions.X_INDEX].min()), np.int_(
            X[..., IndexPositions.Y_INDEX] - X[..., IndexPositions.Y_INDEX].min())] = X[
            ..., IndexPositions.TOT_INDEX]

        # The shift is the number of pixels a all points are shifted in a certain direction to reduce the size of the 2D image.
        # After performing the blob detection and building the centroids the shift has to be added to place the points in the complete image!
        shift_x = X[..., IndexPositions.X_INDEX].min()
        shift_y = X[..., IndexPositions.Y_INDEX].min()

        return image_2D, shift_x, shift_y

    @staticmethod
    def build_2D_image_adding_overlaps(X):
        """ Builds a 2D image of the cluster by taking the sum of al voxels at the same x and y position. The current implementation is a lot slower, than the methods above!
            But it can be more accurate to calculate it with adding the overlaps."""
        x_max = int(X[..., IndexPositions.X_INDEX].max())
        y_max = int(X[..., IndexPositions.Y_INDEX].max())
        image_2D = np.zeros((x_max, y_max))
        for x in range(x_max):
            for y in range(y_max):
                image_2D[x, y] = \
                X[(X[..., IndexPositions.X_INDEX] == x) & (X[..., IndexPositions.Y_INDEX] == y)][
                    ..., IndexPositions.TOT_INDEX].sum()

        shift_x = 0
        shift_y = 0

        return image_2D, shift_x, shift_y

    @staticmethod
    def build_2D_histogram_complete(X):
        dim = 256
        shift_x = 0
        shift_y = 0
        image_2D, _, _ = np.histogram2d(X[..., ProcessingIn2D.X_INDEX], X[..., ProcessingIn2D.Y_INDEX], bins=dim,
                                        range=[[0, dim], [0, dim]], weights=X[..., ProcessingIn2D.TOT_INDEX])

        return image_2D, shift_x, shift_y

    @staticmethod
    def build_2D_histogram_reduced(X):
        x_min, x_max = int(X[..., IndexPositions.X_INDEX].min()), int(X[..., IndexPositions.X_INDEX].max())
        y_min, y_max = int(X[..., IndexPositions.Y_INDEX].min()), int(X[..., IndexPositions.Y_INDEX].max())
        shift_x = x_min
        shift_y = y_min

        if x_min != x_max and y_min != y_max:
            image_2D, _, _ = np.histogram2d(X[..., ProcessingIn2D.X_INDEX], X[..., ProcessingIn2D.Y_INDEX],
                                            bins=[x_max - x_min, y_max - y_min], range=[[x_min, x_max], [y_min, y_max]],
                                            weights=X[..., ProcessingIn2D.TOT_INDEX])
        else:
            # This can happen if a group/ a cluster does only contain a single voxel.
            # Right now I do not know why this happenes also if min_samples > 1 but it does.
            image_2D = np.array([[X[..., 3]]])


        return image_2D, shift_x, shift_y

    @staticmethod
    def map_2D_points_to_3D(voxels, center_points_2D):
        return np.array(list(itertools.chain.from_iterable([
            ProcessingIn2D.map_2D_point_to_3D(voxels, center_point)
            for center_point in center_points_2D
        ])))[..., :ProcessingIn2D.TOT_INDEX + 1]

    @staticmethod
    def map_2D_point_to_3D(voxels, center_point):
        center_points_3D = voxels[np.logical_and(voxels[..., ProcessingIn2D.X_INDEX] == int(center_point[0]),
                                                 voxels[..., ProcessingIn2D.Y_INDEX] == int(center_point[1]))]
        if len(center_points_3D) > 0:
            center_points_3D[..., ProcessingIn2D.X_INDEX] = center_point[ProcessingIn2D.X_INDEX]
            center_points_3D[..., ProcessingIn2D.Y_INDEX] = center_point[ProcessingIn2D.Y_INDEX]
            return center_points_3D[..., :ProcessingIn2D.TOT_INDEX + 1]
        else:
            return np.array([np.concatenate((center_point[: ProcessingIn2D.Y_INDEX + 1], [voxels[..., ProcessingIn2D.TOF_INDEX].mean(), voxels[..., ProcessingIn2D.TOT_INDEX].max()]))])