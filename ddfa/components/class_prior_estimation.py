import numpy as np
from sklearn.decomposition import NMF

from clustering import *

class ClassPriorEstimationTechnique:
    def get_hyperparameter_dict(self):
        pass

    def estimate_class_prior(self, n_classes, n_domains, input_data, input_domains):
        pass

class ClusterNMFClassPriorEstimation(ClassPriorEstimationTechnique):
    def __init__(self, base_cluster_model, n_discretization):
        self.base_cluster_model = base_cluster_model
        self.n_discretization = n_discretization

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'base_params': self.base_cluster_model.get_hyperparameter_dict(),
            'n_discretization': self.n_discretization
        }

    def estimate_class_prior(self, n_classes, n_domains, input_data, input_domains):
      
        self.base_cluster_model.train_cluster(self.n_discretization, n_domains, input_data, input_domains)
        cluster_labels = self.base_cluster_model.eval_cluster(input_data, input_domains)

        discrete_x = cluster_labels
        n_discrete_x = self.n_discretization
        n_latents = n_classes

        bin_edges = [ np.array(list(range(n_discrete_x + 1))) - 0.5, np.array(list(range(n_domains + 1))) - 0.5 ]
        x_matrix = np.histogram2d(discrete_x, input_domains, bins=bin_edges)[0]

        x_matrix = x_matrix / np.sum(x_matrix, axis=0, keepdims=True)

        self.model = NMF(n_components=n_latents, init='random')
        
        W_new = self.model.fit_transform(x_matrix.T).T
        C_new = self.model.components_.T

        col_sums_C = np.sum(C_new, axis=0, keepdims=True)
        col_sums_C_vec = col_sums_C[0]

        C_new /= col_sums_C
        W_new *= np.tile(np.expand_dims(col_sums_C_vec, axis=1), (1,n_domains))

        W_new /= np.sum(W_new, axis=0, keepdims=True)

        self.p_y_given_d = W_new
        self.p_discrete_x_given_y = C_new

        col_sums_C = np.sum(C_new, axis=0, keepdims=True)
        col_sums_C_vec = col_sums_C[0]

        return self.p_y_given_d