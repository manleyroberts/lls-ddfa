import numpy as np
from faiss import Kmeans

from .experiment_utils import *

class ClusterModel:
    def train_cluster(self, n_classes, n_domains, input_data, input_domains):
        pass

    def eval_cluster(self, input_data, input_domains):
        pass

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self)
        }

class ClusterModelFaissKMeans(ClusterModel):
    def __init__(self, use_gpu=False):
        self.use_gpu = use_gpu

    def train_cluster(self, n_classes, n_domains, input_data, input_domains):

        self.model = Kmeans(d=input_data.shape[1], k=n_classes, niter=100, verbose=False, gpu=self.use_gpu, nredo=5)
        self.model.train(input_data.astype(np.float32))
        
        centroid_distances, cluster_labels = self.model.index.search(input_data.astype(np.float32), 1)
        return cluster_labels[:,0]

    def eval_cluster(self, input_data, input_domains):
        centroid_distances, cluster_labels = self.model.index.search(input_data.astype(np.float32), 1)
        return cluster_labels[:,0]
