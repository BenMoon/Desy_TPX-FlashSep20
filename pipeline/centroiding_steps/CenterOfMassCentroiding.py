import numpy as np
import scipy.ndimage as nd

from pipeline.PipelineStep import PipelineStep


class CenterOfMassCentroiding(PipelineStep):

    def perform(self, data):
        if data.shape[0] <= 0:
            return np.empty((0, 4))
        else:
            x = data[..., PipelineStep.X_INDEX]
            y = data[..., PipelineStep.Y_INDEX]
            tof = data[..., PipelineStep.TOF_INDEX]
            tot = data[..., PipelineStep.TOT_INDEX]
            labels = data[..., PipelineStep.LABEL_INDEX]

            label_index = np.unique(labels)
            tot_max = np.array(nd.maximum_position(tot, labels=labels, index=label_index)).flatten()

            tot_sum = nd.sum(tot, labels=labels, index=label_index)
            cluster_x = np.array(nd.sum(x * tot, labels=labels, index=label_index) / tot_sum).flatten()
            cluster_y = np.array(nd.sum(y * tot, labels=labels, index=label_index) / tot_sum).flatten()
            cluster_tof = np.array(nd.sum(tof * tot, labels=labels, index=label_index) / tot_sum).flatten()  # timewalk
            cluster_tot = tot[tot_max]

            return np.column_stack((cluster_x, cluster_y, cluster_tof, cluster_tot))