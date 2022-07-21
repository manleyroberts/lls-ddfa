'''
Implementation: Pranav Mani, Manley Roberts
'''

import copy

import torch
from torch import nn

import numpy as np
import wandb
from models.resnet import BasicBlockWithDropout, ResNet

from domain_discriminator_interface import *

from experiment_utils import *


class VanillaDomainDiscriminatorModel(DomainDiscriminator):
    
    def __init__(self, device, lr, exp_lr_gamma, epochs, batch_size, n_classes, n_domains, eval_ps, class_prior):
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.model = self.build_model().to(device)
        self.epochs = epochs
        self.lr = lr
        self.gamma = exp_lr_gamma
        self.device = device
        self.batch_size = batch_size
        self.eval_ps = eval_ps
        self.class_prior = class_prior

    def build_model(self):
        pass

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'n_domains': self.n_domains,
            'n_epochs': self.epochs,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'final_task_epoch_interval': self.epoch_interval_to_compute_final_task,
        }

    def fit_discriminator(self, train_data, valid_data, train_domains, valid_domains):
        loss_fn = nn.CrossEntropyLoss()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)


        batch_size = self.batch_size

        self.model.train()

        best_epoch = 0
        best_model = None
        best_valid_loss = None

        for epoch in range(self.epochs):
            self.model.train()

            n_correct = 0
            sum_loss = 0
            n_train = 0
            batch_start = 0
            for vec in train_data:
                if isinstance(vec, dict):
                    batch, _ = vec['image'], vec['target']
                else:
                    batch, _ = vec
                # batch = train_data[batch_start:batch_start + batch_size]
                batch = batch.to(self.device)
                labels = train_domains[batch_start:batch_start + batch_size]

                optimizer.zero_grad()
                logits = self.model(batch)
                loss = loss_fn(logits, labels)
                
                sum_loss += loss.detach().cpu().numpy()
                loss.backward()
                optimizer.step()

                n_correct_batch = len(torch.nonzero(torch.argmax(logits, dim=1) == labels))
                n_correct += n_correct_batch
                n_train += batch.shape[0]
                batch_start += batch.shape[0]
            train_acc = n_correct / n_train
            train_loss = sum_loss / n_train

            self.model.eval()

            n_correct = 0
            sum_loss = 0
            n_valid = 0
            batch_start = 0
            for vec in valid_data:
                if isinstance(vec, dict):
                    batch, _ = vec['image'], vec['target']
                else:
                    batch, _ = vec
                # batch = valid_data[batch_start:batch_start + batch_size]
                batch = batch.to(self.device)
                labels = valid_domains[batch_start:batch_start + batch_size]

                with torch.no_grad():

                    logits = self.model(batch)
                    loss = loss_fn(logits, labels)
                    
                    sum_loss += loss.detach().cpu().numpy()

                n_correct_batch = len(torch.nonzero(torch.argmax(logits, dim=1) == labels))
                n_correct += n_correct_batch
                n_valid += batch.shape[0]
                batch_start += batch.shape[0]
            valid_acc = n_correct / n_valid
            valid_loss = sum_loss / n_valid

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)
                best_valid_loss = valid_loss
                
            wandb.log({
                'train_domain_discriminator_accuracy': train_acc,
                'train_domain_discriminator_loss':     train_loss,
                'valid_domain_discriminator_accuracy': valid_acc,
                'valid_domain_discriminator_loss':    valid_loss,
                'epoch': epoch,
                'best_epoch': best_epoch
            })


            scheduler.step()

        # Preserve best model on test dataset
        self.model = best_model

    def get_features(self, data):
        eval_probs_list =[]
        softmax = nn.Softmax().to(self.device)
        self.model.eval()
        corr_labels = []
        for vec in data:
            if isinstance(vec, dict):
                batch, _ = vec['image'], vec['target']
            else:
                batch, _ = vec
            corr_labels.append(_)
            batch = batch.to(self.device)
            with torch.no_grad():
                eval_logits = self.model(batch)
                eval_probs = softmax(eval_logits).cpu().numpy()
            eval_probs_list.append(eval_probs)

        self.corr_labels = np.concatenate(corr_labels, axis=0)

        return np.concatenate(eval_probs_list, axis=0)

class CIFAR10PytorchCifar(VanillaDomainDiscriminatorModel):

    def build_model(self):
        # Dropout variant of Resnet34
        return ResNet(BasicBlockWithDropout, [3, 4, 6, 3], num_classes=self.n_domains)
