import torch
import numpy as np

from experiment_utils import *

class DomainClassPriorMatrix:
    
    def get_class_priors(self):
        pass

    def get_hyperparameter_dict(self):
        pass

class PremadeClassPriorMatrix(DomainClassPriorMatrix):
    
    def __init__(self, n_classes, n_domains, assignment_matrix, min_train_num, min_test_num, min_valid_num):

        self.n_classes = n_classes
        self.n_domains = n_domains

        self.class_priors = assignment_matrix
        self.condition_number = np.linalg.cond(assignment_matrix)
        self.domain_relative_sizes = (np.ones(n_domains) / n_domains)

        fraction_needed_class = (self.domain_relative_sizes.T @ self.class_priors)
        max_fraction_needed = max(fraction_needed_class)

        assignment_scaler_train = min_train_num / max_fraction_needed
        assignment_scaler_test  = min_test_num / max_fraction_needed
        assignment_scaler_valid = min_valid_num / max_fraction_needed

        self.class_domain_assignment_matrix_train = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_train))
        self.class_domain_assignment_matrix_test  = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_test))
        self.class_domain_assignment_matrix_valid = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_valid))

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'n_classes': self.n_classes,
            'n_domains': self.n_domains,
            'condition_number': self.condition_number,
            'matrices': {
                'class_prior': self.class_priors,
                'class_domain_assignment_matrix_train': self.class_domain_assignment_matrix_train,
                'class_domain_assignment_matrix_test': self.class_domain_assignment_matrix_test,
                'class_domain_assignment_matrix_valid': self.class_domain_assignment_matrix_valid
            },
            'n_samples_train': torch.sum(self.class_domain_assignment_matrix_train),
            'n_samples_test': torch.sum(self.class_domain_assignment_matrix_test),
            'n_samples_valid': torch.sum(self.class_domain_assignment_matrix_valid),
        }

class RandomDomainClassPriorMatrix(DomainClassPriorMatrix):

    def __init__(self, n_classes, n_domains, max_condition_number, random_seed, class_prior_alpha, min_train_num, min_test_num, min_valid_num):

        self.n_classes = n_classes
        self.n_domains = n_domains
        self.random_seed = random_seed

        self.class_prior_alpha = class_prior_alpha

        self.class_priors, self.condition_number = self.generate_class_priors(n_classes, n_domains, max_condition_number, random_seed, class_prior_alpha)
        self.domain_relative_sizes = (np.ones(n_domains) / n_domains)

        fraction_needed_class = (self.domain_relative_sizes.T @ self.class_priors)
        max_fraction_needed = max(fraction_needed_class)

        assignment_scaler_train = min_train_num / max_fraction_needed
        assignment_scaler_test  = min_test_num / max_fraction_needed
        assignment_scaler_valid = min_valid_num / max_fraction_needed

        self.class_domain_assignment_matrix_train = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_train))
        self.class_domain_assignment_matrix_test  = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_test))
        self.class_domain_assignment_matrix_valid = torch.Tensor(np.floor(self.class_priors * self.domain_relative_sizes.reshape(n_domains, 1) * assignment_scaler_valid))

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'n_classes': self.n_classes,
            'n_domains': self.n_domains,
            'random_seed': self.random_seed,
            'condition_number': self.condition_number,
            'matrices': {
                'class_prior': self.class_priors,
                'class_domain_assignment_matrix_train': self.class_domain_assignment_matrix_train,
                'class_domain_assignment_matrix_test': self.class_domain_assignment_matrix_test,
                'class_domain_assignment_matrix_valid': self.class_domain_assignment_matrix_valid
            },
            'n_samples_train': torch.sum(self.class_domain_assignment_matrix_train),
            'n_samples_test': torch.sum(self.class_domain_assignment_matrix_test),
            'n_samples_valid': torch.sum(self.class_domain_assignment_matrix_valid),
            'alpha': self.class_prior_alpha
        }

    def generate_class_priors(self, n_classes, n_domains, max_condition_number, random_seed, class_prior_alpha):
        if random_seed is not None:
            np.random.seed(random_seed)
        condition = max_condition_number + 1
        while condition > max_condition_number:

            class_prior_alpha_vec = class_prior_alpha * (np.ones(n_classes) / n_classes)
            class_priors = np.random.dirichlet(class_prior_alpha_vec, n_domains) 
            condition = np.linalg.cond(class_priors)

        class_priors_adjusted = np.where(class_priors < 1e-7, 0, class_priors)
        class_priors_adjusted = class_priors_adjusted / np.sum(class_priors_adjusted, axis=1, keepdims=True)

        class_priors = class_priors_adjusted

        return class_priors, condition