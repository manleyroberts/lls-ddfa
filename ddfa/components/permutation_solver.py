'''
Implementation: Pranav Mani, Manley Roberts
'''

import numpy as np
from scipy.optimize import linear_sum_assignment
from .experiment_utils import *

class PermutationSolver:

    def get_best_permutation(self, label_output, true_classes):
        pass

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self)
        }

class ScipyOptimizeLinearSumPermutationSolver(PermutationSolver):
    def get_best_permutation(self, label_output, true_classes):
        n_classes = max(len(np.unique(label_output)), len(np.unique(true_classes)))
        bin_edges = [ np.array(list(range(n_classes + 1))) - 0.5, np.array(list(range(n_classes + 1))) - 0.5 ]
        cost_matrix = np.histogram2d(label_output, true_classes, bins=bin_edges)[0]
        row_ind, col_ind = linear_sum_assignment(cost_matrix, maximize=True)

        assignment_sorted_col = sorted(zip(row_ind, col_ind), key=lambda pair: pair[0])

        best_class_ordering = [a[1] for a in assignment_sorted_col]

        permuted_labels = np.zeros_like(label_output)
        for class_label, new_class_label in enumerate(best_class_ordering):
            permuted_labels[np.where( label_output == class_label )[0]] = new_class_label

        self.best_class_ordering = best_class_ordering

        return permuted_labels
