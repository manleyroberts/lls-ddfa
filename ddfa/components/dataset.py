
import torch
import numpy as np
import torchvision
from torchvision import transforms

from .experiment_utils import *

"""
Author: Wouter Van Gansbeke, Simon Vandenhende
Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)

Modifications by Pranav Mani and Manley Roberts
"""
import torchvision.datasets as datasets
import torch.utils.data as data

from glob import glob
import os
from PIL import Image

class ImageNet(datasets.ImageFolder):
    def __init__(self, root, split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, 'split',
                                         transform=None))
        self.transform = transform 
        self.split = split
        self.resize = transforms.Resize(256)
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img)

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index}}

        return out

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img


class ImageNetSubset(data.Dataset):
    def __init__(self, subset_file, root, split='train', 
                    transform=None):
        super(ImageNetSubset, self).__init__()

        self.root = os.path.join(root, split)
        self.transform = transform
        self.split = split

        # Read the subset of classes to include (sorted)
        with open(subset_file, 'r') as f:
            result = f.read().splitlines()
        subdirs, class_names = [], []
        for line in result:
            subdir, class_name = line.split(' ', 1)
            subdirs.append(subdir)
            class_names.append(class_name)

        # Gather the files (sorted)
        imgs = []
        for i, subdir in enumerate(subdirs):
            subdir_path = os.path.join(self.root, subdir)
            files = sorted(glob(os.path.join(self.root, subdir, '*.JPEG')))
            for f in files:
                imgs.append((f, i)) 
        self.imgs = imgs 
        self.classes = class_names
    
	# Resize
        self.resize = transforms.Resize(256)

    def get_image(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        img = self.resize(img) 
        return img

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')
        im_size = img.size
        img = self.resize(img) 
        class_name = self.classes[target]

        if self.transform is not None:
            img = self.transform(img)

        out = {'image': img, 'target': target, 'meta': {'im_size': im_size, 'index': index, 'class_name': class_name}}

        return out

"""
Implementation by Pranav Mani and Manley Roberts
"""

class Dataset:
    def __init__(self, data_root, dataset_seed, batch_size=32):
        train_data, valid_data, test_data, train_loader, valid_loader, test_loader, n_train, n_test, n_valid, data_dims, total_pixels_per_image, n_classes = self.get_data(data_root, batch_size, dataset_seed=dataset_seed)
        self.train_data = train_data
        self.valid_data = valid_data
        self.test_data = test_data
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.n_train = n_train
        self.n_test = n_test
        self.n_valid = n_valid
        self.data_dims = data_dims
        self.total_pixels_per_image = total_pixels_per_image
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.preprocess_data()
        self.dataset_seed = dataset_seed

    def get_hyperparameter_dict(self):
        return {
            'n_classes': self.n_classes,
            'data_dims': self.data_dims,
            'n_train': self.n_train,
            'n_test': self.n_test,
            'n_valid': self.n_valid,
            'dataset_seed': self.dataset_seed,
            'name': get_name(self)
        }

    def get_data(self, data_root, batch_size, dataset_seed):
        pass

    def preprocess_data(self):
        train_label_concatenate = torch.zeros(int(self.n_train), 2)
        valid_label_concatenate = torch.zeros(int(self.n_valid), 2)
        test_label_concatenate  = torch.zeros(int(self.n_test), 2)

        loaders = [
            self.train_loader,
            self.valid_loader,
            self.test_loader
        ]

        label_concats = [
            train_label_concatenate,
            valid_label_concatenate,
            test_label_concatenate
        ]

        for loader, label_concatenate in zip(loaders, label_concats):
            b_index = 0

            for vec in loader:
                if isinstance(vec, dict):
                    data, label = vec['image'], vec['target']
                else:
                    data, label = vec
                label_concatenate[b_index :b_index + label.shape[0], 0] = label
                b_index += label.shape[0]

        train_histogram = np.histogram(train_label_concatenate[:,0].cpu().numpy(), bins=self.n_classes)
        min_train_num = min(train_histogram[0])

        valid_histogram = np.histogram(valid_label_concatenate[:,0].cpu().numpy(), bins=self.n_classes)
        min_valid_num = min(valid_histogram[0])

        test_histogram = np.histogram(test_label_concatenate[:,0].cpu().numpy(), bins=self.n_classes)
        min_test_num = min(test_histogram[0])

        self.train_label_concatenate = train_label_concatenate

        self.train_histogram = train_histogram
        self.min_train_num = min_train_num

        self.test_label_concatenate = test_label_concatenate

        self.test_histogram = test_histogram
        self.min_test_num = min_test_num

        self.valid_label_concatenate = valid_label_concatenate

        self.valid_histogram = valid_histogram
        self.min_valid_num = min_valid_num

class ImageNet50(Dataset):
    def get_data(self, data_root, batch_size, dataset_seed):


        n_classes = 50
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        train_base = ImageNetSubset(subset_file='data_utils/imagenet_50.txt', root=data_root, split='train', transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                    ]))

        n_train = len(train_base)

        n_valid = int(0.1 * n_train)
        n_train = n_train - n_valid

        data_dims = (3,256,256)
        total_pixels_per_image  = 3 * 256 * 256

        train_data, _ = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )
        train_base = ImageNetSubset(subset_file='data_utils/imagenet_50.txt', root=data_root, split='train', transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                    ]))

        _, valid_data = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )

        test_data = ImageNetSubset(subset_file='data_utils/imagenet_50.txt', root=data_root, split='val', transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean, std),
                                    ]))

        n_test = len(test_data)
       
        train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data)).tolist())
        valid_data = torch.utils.data.Subset(valid_data, torch.randperm(len(valid_data)).tolist())
        test_data  = torch.utils.data.Subset(test_data,  torch.randperm(len(test_data)).tolist())


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        
        return train_data, valid_data, test_data, train_loader, valid_loader, test_loader, n_train, n_test, n_valid, data_dims, total_pixels_per_image, n_classes

class CIFAR3(Dataset):
    def get_data(self, data_root, batch_size, dataset_seed):

        n_train = 50000
        n_test = 10000

        n_valid = int(0.1 * n_train)
        n_train = n_train - n_valid

        data_dims = (3,32,32)
        total_pixels_per_image  = 3 * 32 * 32

        n_classes = 3

        cifar_10_mean = [
            0.4914,
            0.4822,
            0.4465
        ]

        cifar_10_std = [ 
            0.2023,
            0.1994,
            0.2010
        ]

        train_base = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.Compose([
                                    transforms.RandomResizedCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_10_mean, std=cifar_10_std)
                                    ]))
        train_data, _ = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )
        train_base = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.Compose([
                                    transforms.CenterCrop(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_10_mean, std=cifar_10_std)
                                    ]))
        _, valid_data = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )

        print(train_data.dataset.transform, valid_data.dataset.transform)

        test_data = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    transforms.CenterCrop(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_10_mean, std=cifar_10_std),
                                    ]))

        indices_of_first_3 = []
        for i in range(n_train):
            if train_data[i][1] < 3:
                indices_of_first_3.append(i)
        train_data = torch.utils.data.Subset(train_data, indices_of_first_3)
        indices_of_first_3 = []
        for i in range(n_valid):
            if valid_data[i][1] < 3:
                indices_of_first_3.append(i)
        valid_data = torch.utils.data.Subset(valid_data, indices_of_first_3)
        indices_of_first_3 = []
        for i in range(n_test):
            if test_data[i][1] < 3:
                indices_of_first_3.append(i)
        test_data = torch.utils.data.Subset(test_data, indices_of_first_3)
            

        train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data)).tolist())
        valid_data = torch.utils.data.Subset(valid_data, torch.randperm(len(valid_data)).tolist())
        test_data  = torch.utils.data.Subset(test_data,  torch.randperm(len(test_data)).tolist())


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        n_train = len(train_data)
        n_valid = len(valid_data)
        n_test = len(test_data)
        
        return train_data, valid_data, test_data, train_loader, valid_loader, test_loader, n_train, n_test, n_valid, data_dims, total_pixels_per_image, n_classes


class CIFAR10(Dataset):
    def get_data(self, data_root, batch_size, dataset_seed):

        n_train = 50000
        n_test = 10000

        n_valid = int(0.1 * n_train)
        n_train = n_train - n_valid

        data_dims = (3,32,32)
        total_pixels_per_image  = 3 * 32 * 32

        n_classes = 10

        cifar_10_mean = [
            0.4914,
            0.4822,
            0.4465
        ]

        cifar_10_std = [ 
            0.2023,
            0.1994,
            0.2010
        ]

        train_base = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.Compose([
                                    transforms.RandomResizedCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_10_mean, std=cifar_10_std)
                                    ]))
        train_data, _ = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )
        train_base = torchvision.datasets.CIFAR10(data_root, train=True, download=True, transform=transforms.Compose([
                                    transforms.CenterCrop(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_10_mean, std=cifar_10_std)
                                    ]))
        _, valid_data = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )

        test_data = torchvision.datasets.CIFAR10(data_root, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    transforms.CenterCrop(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_10_mean, std=cifar_10_std),
                                    ]))
       
        train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data)).tolist())
        valid_data = torch.utils.data.Subset(valid_data, torch.randperm(len(valid_data)).tolist())
        test_data  = torch.utils.data.Subset(test_data,  torch.randperm(len(test_data)).tolist())


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        
        return train_data, valid_data, test_data, train_loader, valid_loader, test_loader, n_train, n_test, n_valid, data_dims, total_pixels_per_image, n_classes

coarse_labels_cifar_20 = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,
                                   3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                                   6, 11,  5, 10,  7,  6, 13, 15,  3, 15, 
                                   0, 11,  1, 10, 12, 14, 16,  9, 11,  5,
                                   5, 19,  8,  8, 15, 13, 14, 17, 18, 10,
                                   16, 4, 17,  4,  2,  0, 17,  4, 18, 17,
                                   10, 3,  2, 12, 12, 16, 12,  1,  9, 19, 
                                   2, 10,  0,  1, 16, 12,  9, 13, 15, 13,
                                  16, 19,  2,  4,  6, 19,  5,  5,  8, 19,
                                  18,  1,  2, 15,  6,  0, 17,  8, 14, 13])

class CIFAR100Coarse(torchvision.datasets.CIFAR100):
    def __init__(self,root,train,download,transform):
        super(CIFAR100Coarse, self).__init__(root, train, download=download, transform=transform)
        self.targets = list(coarse_labels_cifar_20[self.targets])

class CIFAR20(Dataset): 
    def get_data(self, data_root, batch_size, dataset_seed):

        cifar_100_mean = [ 
            0.5071,
            0.4867,
            0.4408
        ]

        cifar_100_std = [ 
            0.2675,
            0.2565,
            0.2761
        ]

        n_train = 50000
        n_test = 10000

        n_valid = int(0.1 * n_train)
        n_train = n_train - n_valid

        data_dims = (3,32,32)
        total_pixels_per_image  = 3 * 32 * 32

        n_classes = 20

        train_base = CIFAR100Coarse(data_root, train=True, download=True, transform=transforms.Compose([
                                    transforms.RandomResizedCrop(32),
                                    transforms.RandomHorizontalFlip(),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_100_mean, std=cifar_100_std)
                                    ]))
        train_data, _ = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )
        train_base = CIFAR100Coarse(data_root, train=True, download=True, transform=transforms.Compose([
                                    transforms.CenterCrop(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_100_mean, std=cifar_100_std),
                                    ]))
        _, valid_data = torch.utils.data.random_split(
            train_base, [n_train, n_valid], generator=torch.Generator().manual_seed(dataset_seed)
        )

        test_data = CIFAR100Coarse(data_root, train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    transforms.CenterCrop(32),
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(mean=cifar_100_mean, std=cifar_100_std),
                                    ]))

        train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data)).tolist())
        valid_data = torch.utils.data.Subset(valid_data, torch.randperm(len(valid_data)).tolist())
        test_data  = torch.utils.data.Subset(test_data,  torch.randperm(len(test_data)).tolist())


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        
        return train_data, valid_data, test_data, train_loader, valid_loader, test_loader, n_train, n_test, n_valid, data_dims, total_pixels_per_image, n_classes

class FieldGuide2(Dataset):
    def get_data(self, data_root, batch_size, dataset_seed):
        data_dims = (3,224,224)
        total_pixels_per_image  = 3 * 224 * 224

        n_classes = 2

        fieldguide2_directory = data_root
        fg2_mean = [0.4923402, 0.49349242, 0.35889822]
        fg2_std  = [0.25436208, 0.24372408, 0.25995356]
        train_data = torchvision.datasets.ImageFolder(fieldguide2_directory + 'train',
                transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(fg2_mean, fg2_std),
                                    ]))

        valid_data = torchvision.datasets.ImageFolder(fieldguide2_directory + 'valid',
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=fg2_mean,std=fg2_std)
                ]))


        test_data = torchvision.datasets.ImageFolder(fieldguide2_directory + 'test',
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=fg2_mean,std=fg2_std)
                ]))

        train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data)).tolist())
        valid_data = torch.utils.data.Subset(valid_data, torch.randperm(len(valid_data)).tolist())
        test_data  = torch.utils.data.Subset(test_data,  torch.randperm(len(test_data)).tolist())


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        n_train = len(train_data)
        n_valid = len(valid_data)
        n_test = len(test_data)


        
        return train_data, valid_data, test_data, train_loader, valid_loader, test_loader, n_train, n_test, n_valid, data_dims, total_pixels_per_image, n_classes

class FieldGuide28(Dataset):

    def get_data(self, data_root, batch_size, dataset_seed):
        data_dims = (3,224,224)
        total_pixels_per_image  = 3 * 224 * 224

        n_classes = 28

        fieldguide28_directory = data_root
        fg2_mean = [0.4923402, 0.49349242, 0.35889822]
        fg2_std  = [0.25436208, 0.24372408, 0.25995356]
        train_data = torchvision.datasets.ImageFolder(fieldguide28_directory + 'train',
                transform=transforms.Compose([
                                        transforms.Resize(256),
                                        transforms.RandomCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize(fg2_mean, fg2_std),
                                    ]))

        valid_data = torchvision.datasets.ImageFolder(fieldguide28_directory + 'valid',
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=fg2_mean,std=fg2_std)
                ]))


        test_data = torchvision.datasets.ImageFolder(fieldguide28_directory + 'test',
                transform=transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=fg2_mean,std=fg2_std)
                ]))

        train_data = torch.utils.data.Subset(train_data, torch.randperm(len(train_data)).tolist())
        valid_data = torch.utils.data.Subset(valid_data, torch.randperm(len(valid_data)).tolist())
        test_data  = torch.utils.data.Subset(test_data,  torch.randperm(len(test_data)).tolist())


        train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size, shuffle=False)
        test_loader  = torch.utils.data.DataLoader(test_data,  batch_size=batch_size, shuffle=False)

        n_train = len(train_data)
        n_valid = len(valid_data)
        n_test = len(test_data)
        
        return train_data, valid_data, test_data, train_loader, valid_loader, test_loader, n_train, n_test, n_valid, data_dims, total_pixels_per_image, n_classes
