'''
Implementation: Pranav Mani, Manley Roberts
'''

import copy

import torch
from torch import nn
import torchvision

import numpy as np
import wandb

from models.resnet import  *
from scan_model_definitions import *
from domain_discriminator_interface import *

from experiment_utils import *

class DomainDiscriminatorSCAN(DomainDiscriminator):
    
    def __init__(self, device, lr, exp_lr_gamma, epochs, batch_size, n_classes, n_domains,load_path=None, eval_ps=None, class_prior=None, dropout=0, limit_gradient_flow=False, use_scheduler='ExponentialLR', baseline_load_path=None):
        
        self.n_classes = n_classes
        self.n_domains = n_domains
        self.load_path = load_path
        
        try:
            assert(dropout in [0,1])
        except: 
            dropout = 1

        self.dropout = dropout
        self.limit_gradient_flow = limit_gradient_flow

        self.model = self.build_model().to(device)
        self.epochs = epochs
        self.lr = lr
        self.gamma = exp_lr_gamma
        self.device = device
        self.batch_size = batch_size

        self.eval_ps = eval_ps
        self.class_prior = class_prior
        assert(use_scheduler in ['ReduceLROnPlateau' , 'ExponentialLR'])
        self.use_scheduler = use_scheduler

        self.baseline_load_path = baseline_load_path

    def build_model(self):
        pass

    def baseline_acc(self, test_data, test_labels):
        pass

    def get_hyperparameter_dict(self):
        return {
            'name': get_name(self),
            'n_domains': self.n_domains,
            'n_epochs': self.epochs,
            'lr': self.lr,
            'gamma': self.gamma,
            'batch_size': self.batch_size,
            'dropout' : self.dropout,
            'limit_gradient_flow' : self.limit_gradient_flow,
            'use_scheduler' : self.use_scheduler,
            'load_path': self.load_path,
            'baseline_load_path': self.baseline_load_path
        }

    def fit_discriminator(self, train_data, valid_data, train_domains, valid_domains):

        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        if(self.use_scheduler == 'ExponentialLR'):
            scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.gamma)
        elif(self.use_scheduler == 'ReduceLROnPlateau'):
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',factor=0.01,patience=5)

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

            if(self.use_scheduler == 'ExponentialLR'):
                scheduler.step()
            elif(self.use_scheduler == 'ReduceLROnPlateau'):
                scheduler.step(valid_loss) 

            if best_valid_loss is None or valid_loss < best_valid_loss:
                best_epoch = epoch
                best_model = copy.deepcopy(self.model)
                best_valid_loss = valid_loss
            
            wandb.log({
                'epoch':                                epoch,
                'train_domain_discriminator_accuracy':  train_acc,
                'train_domain_discriminator_loss':      train_loss,
                'valid_domain_discriminator_accuracy':  valid_acc,
                'valid_domain_discriminator_loss':      valid_loss,
                'best_epoch':                           best_epoch,
                'scan_alone_test_acc':                  scan_alone_test_acc,
                'scan_alone_reconstruction_error_L1':   scan_alone_reconstruction_error_L1,
                'scan_reconstructed_p_y_given_d':       scan_reconstructed_p_y_given_d
            })

        # Preserve best model on test dataset
        self.model = best_model

    def get_features(self, data):
        # batch_size = 32
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
            # batch = data[i:i+batch_size]
            with torch.no_grad():
                eval_logits = self.model(batch)
                eval_probs = softmax(eval_logits).cpu().numpy()
            eval_probs_list.append(eval_probs)

        self.corr_labels = np.concatenate(corr_labels, axis=0)

        return np.concatenate(eval_probs_list, axis=0)


class scan_pretext(DomainDiscriminatorSCAN):
    def baseline_acc(self, test_data, test_labels, test_domains, true_p_y_d):
        load_path = self.baseline_load_path
        backbone = ResNet18_for_SCAN()
        model = ClusteringModel(backbone,nclusters=self.n_classes,nheads=1).to(self.device) 
        state_dict = torch.load(load_path)
        if 'model' in state_dict:
            state_dict = state_dict['model'] 
        model.load_state_dict(state_dict,strict=True)
        scan_alone_test_acc, p_y_d_err, scan_p_y_d  = model_evaluate(model, test_data, true_p_y_d, self.eval_ps, self.device)

        return scan_alone_test_acc, p_y_d_err, scan_p_y_d

    def build_model(self):
        num_classes = self.n_domains
        dropout = self.dropout
        limit_gradient_flow = self.limit_gradient_flow
        if(dropout == 0):
            obj =  ResNetwithInitialisations(BasicBlock,[2,2,2,2],num_classes=num_classes) 
        else: 
            obj = ResNetwithInitialisations(BasicBlockWithDropout,[2,2,2,2],num_classes=num_classes)
            
        load_path = self.load_path
        state = torch.load(load_path)
        from collections import OrderedDict
        renamed_state = OrderedDict()
        for key in state.keys(): 
            renamed_state[key.replace('backbone.','')] = state[key] 
        
        obj.load_state_dict(renamed_state, strict=False)

        if(limit_gradient_flow == True):
            print("Limiting gradient flow")
            for module in obj.modules(): 
                if(isinstance(module,nn.Linear)):
                    for param in module.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in module.parameters():
                        param.requires_grad = False 

        return obj


class scan_scan(DomainDiscriminatorSCAN):
    def baseline_acc(self, test_data, test_labels, test_domains, true_p_y_d):
        load_path = self.baseline_load_path
        backbone = ResNet18_for_SCAN()
        model = ClusteringModel(backbone,nclusters=self.n_classes,nheads=1).to(self.device) 
        state_dict = torch.load(load_path)
        if 'model' in state_dict:
            state_dict = state_dict['model']

        model_dict = model.state_dict()

        # Fiter out unneccessary keys
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

        scan_alone_test_acc, p_y_d_err, scan_p_y_d  = model_evaluate(model, test_data, true_p_y_d, self.eval_ps, self.device)

        return scan_alone_test_acc, p_y_d_err, scan_p_y_d

    def build_model(self): 
        num_classes = self.n_domains
        dropout = self.dropout
        limit_gradient_flow = self.limit_gradient_flow
        if(dropout == 0):
            obj =  ResNetwithInitialisations(BasicBlock,[2,2,2,2],num_classes=num_classes) 
        else: 
            obj = ResNetwithInitialisations(BasicBlockWithDropout,[2,2,2,2],num_classes=num_classes)
         
        load_path = self.load_path
        state_full = torch.load(load_path)
        state_dict = state_full['model']
        from collections import OrderedDict
        renamed_state = OrderedDict()
        for key in state_dict.keys(): 
            renamed_state[key.replace('backbone.','')] = state_dict[key]

        obj.load_state_dict(renamed_state,strict=False)

        if(limit_gradient_flow==True):
            print("Limiting gradient flow")
            for module in obj.modules(): 
                if(isinstance(module,nn.Linear)):
                    for param in module.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in module.parameters():
                        param.requires_grad = False 
        return obj 


class scan_scan_imagenet(DomainDiscriminatorSCAN):
    def baseline_acc(self, test_data, test_labels, test_domains, true_p_y_d):
        
        load_path = self.baseline_load_path
        obj  = torchvision.models.__dict__['resnet50']()
        obj.fc = nn.Identity()
        backbone = {'backbone': obj, 'dim':2048}
        model = ClusteringModel(backbone,nclusters=self.n_classes,nheads=1).to(self.device) 
        state_dict = torch.load(load_path)
        if 'model' in state_dict:
            state_dict = state_dict['model'] 
        missing = model.load_state_dict(state_dict,strict=False)
        scan_alone_test_acc, p_y_d_err, scan_p_y_d  = model_evaluate(model, test_data, true_p_y_d, self.eval_ps, self.device)

        return scan_alone_test_acc, p_y_d_err, scan_p_y_d

    def build_model(self): 
        num_classes = self.n_domains
        dropout = self.dropout
        limit_gradient_flow = self.limit_gradient_flow

        if(dropout != 0):
            print("Provision to add dropout has not yet been implemented") 
            raise ValueError 

        obj =  torchvision.models.__dict__['resnet50']()
        obj.fc = nn.Linear(2048,num_classes,bias=True) 
         
        load_path = self.load_path
        state_full = torch.load(load_path)
        state_dict = state_full['model']
        from collections import OrderedDict
        renamed_state = OrderedDict()
        for key in state_dict.keys(): 
            renamed_state[key.replace('backbone.','')] = state_dict[key]

        obj.load_state_dict(renamed_state,strict=False)

        if(limit_gradient_flow==True):
            print("Limiting gradient flow")
            for module in obj.modules(): 
                if(isinstance(module,nn.Linear)):
                    for param in module.parameters(): 
                        param.requires_grad = True 
                else: 
                    for param in module.parameters():
                        param.requires_grad = False 
        return obj   

        