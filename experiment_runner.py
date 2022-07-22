import os

import argparse
from pyparsing import alphanums
import yaml

parser = argparse.ArgumentParser(description='Pass exactly one argument: the path to the experiment config yaml file.')
parser.add_argument('--dataset_config_path', type=str, default='dataset_config.yml',
                    help='The path to the experiment config yaml file')
args = parser.parse_args()

with open(args.dataset_config_path) as f:
    experiment_config = yaml.full_load(f)

os.environ["CUDA_VISIBLE_DEVICES"]=str(experiment_config['gpu'])    

import torch
import wandb
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print('Using device: ', device)

from ddfa.components.dataset import *
from ddfa.components.permutation_solver import *
from ddfa.components.experiment_utils import *
from ddfa.components.domain_discriminator.scan_model_definitions import *
from ddfa.components.domain_discriminator.domain_discriminator_scan import * 
from ddfa.components.experiment_framework import *

for experiment in experiment_config['experiments']:
    dataset_choice  = experiment['dataset_settings']['dataset']
    dataset_seed    = experiment['dataset_settings']['dataset_split_seed']

    dataset_root    = experiment_config['datasets'][dataset_choice]['root_path']

    domains         = experiment['class_prior_generation']['domains']
    alpha           = experiment['class_prior_generation']['alpha']
    max_cond_number = experiment['class_prior_generation']['max_condition_number']
    class_prior_seed= experiment['class_prior_generation']['class_prior_seed']

    use_raw_ddfa    = 'ddfa'        in experiment['approaches']
    use_scan        = 'ddfa_scan'   in experiment['approaches']

    if dataset_choice == 'cifar10':
        dummy_dataset_instance = CIFAR10(42)
        dataset_class = CIFAR10

        scan_ddfa_epochs        = 25
        scan_ddfa_loadpath      = './pretrain/scan_cifar_pretrain/scan_cifar-10.pth.tar'
        scan_ddfa_subclass_name = scan_scan
        baseline_scan_name      = scan_ddfa_loadpath

        ddfa_epochs             = 100
        ddfa_n_discretization   = 30

    elif dataset_choice == 'cifar3':
        dummy_dataset_instance = CIFAR3(42)
        dataset_class = CIFAR3

        scan_ddfa_epochs        = 20
        scan_ddfa_loadpath      = './pretrain/scan_cifar_pretrain/scan_cifar-10.pth.tar'
        scan_ddfa_subclass_name = scan_scan

        baseline_scan_name      = scan_ddfa_loadpath

        ddfa_epochs             = 100
        ddfa_n_discretization   = 10

    elif dataset_choice == 'cifar20':
        dummy_dataset_instance = CIFAR20(42)
        dataset_class = CIFAR20

        scan_ddfa_epochs        = 25
        scan_ddfa_loadpath      = './pretrain/scan_cifar_pretrain/scan_cifar-20.pth.tar'
        scan_ddfa_subclass_name = scan_scan

        baseline_scan_name      = scan_ddfa_loadpath

        ddfa_epochs             = 100
        ddfa_n_discretization   = 60

    elif dataset_choice == 'imagenet':
        dummy_dataset_instance = ImageNet50(42)
        dataset_class = ImageNet50

        scan_ddfa_epochs        = 25
        scan_ddfa_loadpath      = './pretrain/scan_imagenet_pretrain/scan_imagenet_50.pth.tar'
        scan_ddfa_subclass_name = scan_scan_imagenet

        baseline_scan_name      = scan_ddfa_loadpath

    elif dataset_choice == 'fg2':
        dummy_dataset_instance = FieldGuide2(42)
        dataset_class = FieldGuide2

        scan_ddfa_epochs        = 30
        scan_ddfa_loadpath      = './pretrain/scan_fieldguide_pretrain/fieldguide2/pretext/model.pth.tar'
        scan_ddfa_subclass_name = scan_pretext

        # for comparison
        baseline_scan_name      = './pretrain/scan_fieldguide_pretrain/fieldguide2/scan/model.pth.tar'

        ddfa_epochs             = 100
        ddfa_n_discretization   = 10

    elif dataset_choice == 'fg28':
        dummy_dataset_instance = FieldGuide28(42)
        dataset_class = FieldGuide28

        scan_ddfa_epochs        = 60
        scan_ddfa_loadpath      = './pretrain/scan_fieldguide_pretrain/fieldguide28/pretext/model.pth.tar'
        scan_ddfa_subclass_name = scan_pretext
        # for comparison
        baseline_scan_name      = './pretrain/scan_fieldguide_pretrain/fieldguide28/scan/model.pth.tar'

        ddfa_epochs             = 100
        ddfa_n_discretization   = 84



        runs = []

        class_prior = RandomDomainClassPriorMatrix(
            n_classes = dummy_dataset_instance.n_classes, 
            n_domains = domains, 
            max_condition_number = max_cond_number, 
            random_seed = class_prior_seed, 
            class_prior_alpha = alpha, 
            min_train_num = dummy_dataset_instance.min_train_num,
            min_test_num = dummy_dataset_instance.min_test_num, 
            min_valid_num = dummy_dataset_instance.min_valid_num
        )
        
        dataset_instance = dataset_class(dataset_seed=dataset_seed)

        if use_scan:
            # Add scan main run 
            runs.append({
                'n_domains': domains,
                'class_prior': class_prior,
                'class_prior_estimator': ClusterNMFClassPriorEstimation(
                    base_cluster_model = ClusterModelFaissKMeans(use_gpu=False),
                    n_discretization = dummy_dataset_instance.n_classes,
                ),
                'dataset': dataset_instance,
                'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
                'discriminator': scan_ddfa_subclass_name(
                        device,
                        lr = 0.00001,
                        exp_lr_gamma = 0.97,
                        epochs = scan_ddfa_epochs,
                        batch_size = 32,
                        n_classes= class_prior.n_classes,
                        n_domains = domains,
                        load_path= scan_ddfa_loadpath,                    
                        eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                        class_prior = class_prior,
                        epoch_interval_to_compute_final_task=100,
                        dropout = 0,
                        limit_gradient_flow=False,
                        use_scheduler = 'ExponentialLR',
                        baseline_load_path=baseline_scan_name
                ),
                'alpha': alpha
            })


        if use_raw_ddfa:        
            # Add Domain Discriminator run
            runs.append({
                'n_domains': domains,
                'class_prior': class_prior,
                'class_prior_estimator': ClusterNMFClassPriorEstimation(
                    base_cluster_model = ClusterModelFaissKMeans(use_gpu=False),
                    n_discretization = ddfa_n_discretization,
                ),
                'dataset': dataset_instance,
                'permutation_solver': ScipyOptimizeLinearSumPermutationSolver(),
                'discriminator': CIFAR10PytorchCifar(
                # 'extractor': CIFAR10PytorchCifar(
                    device = device,
                    lr = 0.001,
                    exp_lr_gamma = 0.97,

                    epochs = ddfa_epochs,

                    batch_size = 32,
                    n_classes = class_prior.n_classes,
                    n_domains = domains,
                    eval_ps = ScipyOptimizeLinearSumPermutationSolver(),
                    class_prior = class_prior
                ),
            })

        for r in runs:

            n_domains               = r['n_domains']
            class_prior             = r['class_prior']
            class_prior_estimator   = r['class_prior_estimator']
            permutation_solver      = r['permutation_solver']
            discriminator           = r['discriminator']
            dataset_instance        = r['dataset']

            config = {
                component_name : component.get_hyperparameter_dict()
                for component_name, component in [
                    ('dataset', dataset_instance),
                    ('class_prior', class_prior),
                    ('class_prior_estimator', class_prior_estimator),
                    ('permutation_solver', permutation_solver),
                    ('discriminator', discriminator)
                ]
            }

            run = wandb.init(
                entity="latent-label-shift-2022",
                project="latent-label-shift-2022-final",
                reinit=True,
                config=config
            )

            experiment = ExperimentSetup(dataset_instance, class_prior, discriminator, class_prior_estimator, permutation_solver, device, batch_size=32)

            wandb.config.update({"final_best_labels": list(experiment.permuted_labels)})
            wandb.config.update({'test_post_cluster_acc': experiment.test_post_cluster_acc})
            wandb.config.update({'test_post_cluster_p_y_given_d_l1_norm': experiment.test_post_cluster_p_y_given_d_l1_norm})
    
            run.finish()