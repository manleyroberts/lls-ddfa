
import numpy as np
import torch
from sklearn.utils import shuffle

from class_prior import *
from class_prior_estimation import *
from clustering import *
from dataset import *
from experiment_utils import *
from domain_discriminator.domain_discriminator import *
from domain_discriminator.domain_discriminator_interface import *
from domain_discriminator.domain_discriminator_scan import *
from permutation_solver import *

class ExperimentSetup:
    def __init__(self, dataset, domain_class_prior_matrix, domain_discriminator, class_prior_estimator, permutation_solver, device, batch_size):
        self.dataset = dataset
        self.domain_class_prior_matrix = domain_class_prior_matrix
        self.domain_discriminator = domain_discriminator
        self.class_prior_estimator = class_prior_estimator
        self.permutation_solver = permutation_solver
        self.n_domains = self.domain_class_prior_matrix.n_domains
        self.n_classes = self.dataset.n_classes

        class_domain_assignment_matrix_train = self.domain_class_prior_matrix.class_domain_assignment_matrix_train
        class_domain_assignment_matrix_test  = self.domain_class_prior_matrix.class_domain_assignment_matrix_test
        class_domain_assignment_matrix_valid = self.domain_class_prior_matrix.class_domain_assignment_matrix_valid

        data_dims = self.dataset.data_dims

        self.train_data, self.test_data, self.valid_data, self.train_labels, self.test_labels, self.valid_labels = self.format_data(
            class_domain_assignment_matrix_train,
            class_domain_assignment_matrix_test,
            class_domain_assignment_matrix_valid,
            data_dims,
            batch_size
        )

        self.train_labels = self.train_labels.to(device)
        self.test_labels  = self.test_labels.to(device)
        self.valid_labels = self.valid_labels.to(device)

        train_domains = self.train_labels[:,1]
        valid_domains = self.valid_labels[:,1]
        test_domains = self.test_labels[:,1]

        train_labels_only = self.train_labels[:,0]
        valid_labels_only = self.valid_labels[:,0]
        test_labels_only  = self.test_labels[:,0]

        self.domain_discriminator.fit_discriminator(self.train_data, self.valid_data, train_domains, valid_domains, train_labels_only, valid_labels_only)

        # Pre-compute best possible acc, if it's available
        if hasattr(self.domain_discriminator, 'baseline_acc'):
            scan_alone_test_acc, scan_alone_reconstruction_error_L1, scan_reconstructed_p_y_given_d = self.domain_discriminator.baseline_acc(self.test_data, test_labels_only, test_domains, self.domain_class_prior_matrix.class_priors.T)

        # Valid for clustering
        cluster_features_train = self.domain_discriminator.get_features(
           self.train_data
        )
        cluster_features_valid = self.domain_discriminator.get_features(
           self.valid_data
        )
        cluster_features_valid_train = np.concatenate([cluster_features_valid, cluster_features_train], axis=0)
        cluster_features_test = self.domain_discriminator.get_features(
            self.test_data
        )


        # self.true_test_classes = self.test_labels[:,0].cpu().numpy()

        valid_domains = valid_domains.cpu().numpy()
        train_domains = train_domains.cpu().numpy()
        test_domains  = test_domains.cpu().numpy()

        domains_valid_train = np.concatenate([valid_domains, train_domains], axis=0)

        # CLUSTER ON TRAIN + VALID
        p_y_d = class_prior_estimator.estimate_class_prior(
            n_classes=self.n_classes,
            n_domains=self.n_domains,
            input_data=cluster_features_valid_train,
            input_domains=domains_valid_train)

        p_d_x = cluster_features_test#.detach().cpu().numpy()
        _, p_y_x = y_predictions_dd_uniform(p_d_x, p_y_d)
        # domain adjusted
        solved_dd_test_labels, p_y_x_d = y_predictions_dd_balanced(p_y_d, p_y_x, test_domains)
        permuted_labels = self.permutation_solver.get_best_permutation(solved_dd_test_labels, test_labels_only.cpu().numpy())
        self.test_post_cluster_acc = label_accuracy(permuted_labels, test_labels_only.cpu().numpy())

        predicted_class_prior = np.zeros_like(self.clusterer.p_y_given_d)
        permuted_p_y_x_d = np.zeros_like(p_y_x_d)
        best_class_ordering = self.permutation_solver.best_class_ordering                
        for class_label, new_class_label in enumerate(best_class_ordering):
            predicted_class_prior[new_class_label,:] = self.clusterer.p_y_given_d[class_label,:]
            permuted_p_y_x_d[:, new_class_label] = p_y_x_d[:, class_label]

        reconstruction_error_L1_balanced = np.sum(abs(self.domain_class_prior_matrix.class_priors.T - predicted_class_prior))
        self.test_post_cluster_p_y_given_d_l1_norm = reconstruction_error_L1_balanced

    def format_data(self, class_domain_assignment_matrix_train, class_domain_assignment_matrix_test, class_domain_assignment_matrix_valid, data_dims, batch_size):
        train_data, train_labels = self.build_data_label_matrix(self.dataset.train_data, self.dataset.train_label_concatenate, class_domain_assignment_matrix_train, self.dataset.n_train, self.domain_class_prior_matrix.n_domains, data_dims, batch_size)
        test_data, test_labels = self.build_data_label_matrix(self.dataset.test_data, self.dataset.test_label_concatenate, class_domain_assignment_matrix_test, self.dataset.n_test, self.domain_class_prior_matrix.n_domains, data_dims, batch_size)
        valid_data, valid_labels = self.build_data_label_matrix(self.dataset.valid_data, self.dataset.valid_label_concatenate, class_domain_assignment_matrix_valid, self.dataset.n_valid, self.domain_class_prior_matrix.n_domains, data_dims, batch_size)
        return train_data, test_data, valid_data, train_labels, test_labels, valid_labels

    def build_data_label_matrix(self, dataset, label_concatenate, class_domain_assignment_matrix, n_data, n_domains, data_dims, batch_size):
        dims = [int(torch.sum(class_domain_assignment_matrix))] + list(data_dims)
        data = torch.zeros(*dims)
        labels = torch.zeros(int(torch.sum(class_domain_assignment_matrix)) , 2).long()
        done_flag = False
        data_index = 0

        indices_list = []

        copy_class_domain_assignment_matrix = class_domain_assignment_matrix.clone()
        for i in range(int(n_data)):
            if done_flag:
                break
            class_label = label_concatenate[i, 0]
            for domain in range(n_domains):
                still_needed = copy_class_domain_assignment_matrix[domain, int(class_label)]
                if still_needed > 0:
                    copy_class_domain_assignment_matrix[domain, int(class_label)] -= 1
                    label_concatenate[i, 1] = domain

                    indices_list.append(i)
                    labels[data_index]   = label_concatenate[i]
                    
                    if data_index == data.shape[0]:
                        done_flag = True

                    data_index += 1
                    break

        if labels.shape[0] > len(indices_list):
            labels = labels[:len(indices_list)]
        
        indices_list, labels = shuffle(indices_list, labels)

        dataset_subset = torch.utils.data.Subset(dataset, indices_list)
        data = torch.utils.data.DataLoader(dataset_subset, batch_size=batch_size, shuffle=False)

        return data, labels
