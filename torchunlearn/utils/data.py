def get_subloader(given_loader, n_limit):
    if n_limit is None:
        return given_loader

    sub_loader = []
    num = 0
    for item in given_loader:
        sub_loader.append(item)
        if isinstance(item, tuple) or isinstance(item, list):
            batch_size = len(item[0])
        else:
            batch_size = len(item)
        num += batch_size
        if num >= n_limit:
            break
    return sub_loader

import torch
import numpy as np
import torchvision.transforms as transforms
from torchunlearn.utils.datasets import Datasets

def sample_indices(labels, ratio, stratified=True, seed=None):
    np.random.seed(seed)
    if isinstance(labels, torch.Tensor):
        labels = labels.numpy()
    num_samples = int(len(labels) * ratio)
    if stratified:
        label_indices = {}
        for idx, label in enumerate(labels):
            if label not in label_indices:
                label_indices[label] = []
            label_indices[label].append(idx)
        
        sampled_indices = {}
        for label, indices in label_indices.items():
            np.random.seed(seed)
            sampled_indices[label] = np.random.choice(indices, num_samples//len(label_indices.keys()), replace=False)

        return [idx for indices in sampled_indices.values() for idx in indices]
    else:
        np.random.seed(seed)
        return np.random.choice(len(labels), num_samples, replace=False)

class UnlearnDataSetup:
    def __init__(self, data_name, n_classes, mean, std):
        self.data_name = data_name
        self.n_classes = n_classes
        self.mean = mean
        self.std = std
    
    def get_transform(self, train_shuffle_and_transform):
        """Returns the appropriate transform based on whether augmentation is applied."""
        base_transform = []
        if self.data_name == "CIFAR10":
            img_size = 32
            if train_shuffle_and_transform:
                base_transform = [
                    transforms.RandomCrop(img_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            else:
                base_transform = [
                    transforms.Resize((img_size, img_size)),
                ]
        elif self.data_name == "CIFAR100":
            img_size = 32
            if train_shuffle_and_transform:
                base_transform = [
                    transforms.RandomCrop(img_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            else:
                base_transform = [
                    transforms.Resize((img_size, img_size)),
                ]
        elif self.data_name == "TinyImageNet":
            img_size = 64
            if train_shuffle_and_transform:
                base_transform = [
                    transforms.RandomCrop(img_size, padding=4),
                    transforms.RandomHorizontalFlip(),
                ]
            else:
                base_transform = [
                    transforms.Resize((img_size, img_size)),
                ]
        else:
            raise ValueError("Not supported data_name.")
                
        return transforms.Compose(base_transform + [
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])
    
    def get_loaders(self, batch_size, train_shuffle_and_transform=False, drop_last_train=True):
        """Gets dataloaders for standard training."""            
        train_transform = self.get_transform(train_shuffle_and_transform)
        test_transform = self.get_transform(False)
        shuffle_train = train_shuffle_and_transform
        
        data = Datasets(
            data_name=self.data_name, 
            train_transform=train_transform,
            test_transform=test_transform
        )
        train_loader, test_loader = data.get_loader(
            batch_size=batch_size, drop_last_train=drop_last_train, shuffle_train=shuffle_train
        )
        
        return train_loader, test_loader
    
    def get_loaders_for_classwise(self, batch_size, omit_label, 
                                  train_shuffle_and_transform=False, drop_last_train=True):
        """Gets dataloaders for unlearning a specific class."""
        if not isinstance(omit_label, int):
            raise ValueError("Omit Label should be int.")
            
        train_transform = self.get_transform(train_shuffle_and_transform)
        test_transform = self.get_transform(False)
        shuffle_train = train_shuffle_and_transform
        
        retain_data = Datasets(
            data_name=self.data_name, 
            label_filter={i: i for i in range(self.n_classes) if i != omit_label},
            train_transform=train_transform,
            test_transform=test_transform
        )
        retain_train_loader, retain_test_loader = retain_data.get_loader(
            batch_size=batch_size, drop_last_train=drop_last_train, shuffle_train=shuffle_train
        )
        
        forget_data = Datasets(
            data_name=self.data_name, 
            label_filter={omit_label: omit_label},
            train_transform=train_transform,
            test_transform=test_transform
        )
        forget_train_loader, forget_test_loader = forget_data.get_loader(
            batch_size=batch_size, drop_last_train=drop_last_train, shuffle_train=shuffle_train
        )

        train_loaders = {
            "Retain": retain_train_loader,
            "Forget": forget_train_loader,
        }

        test_loaders = {
            "Test": retain_test_loader,
            "Test(forget)": forget_test_loader,  # for exception case
        }
        
        return train_loaders, test_loaders       
    
    def get_loaders_for_rand(self, batch_size, ratio, stratified=True, 
                             train_shuffle_and_transform=False, drop_last_train=True, seed=42):
        """Gets dataloaders for random removal of a ratio of data."""
        if not (0 < ratio < 1):
            raise ValueError("Ratio must be in the range (0,1).")
        
        train_transform = self.get_transform(train_shuffle_and_transform)
        test_transform = self.get_transform(False)
        shuffle_train = train_shuffle_and_transform

        # extract sample indices
        data = Datasets(
            data_name=self.data_name,
            train_transform=train_transform,
            val_transform=test_transform,  # no matter what
            test_transform=test_transform  # no matter what
        )
        labels = torch.tensor(data.train_data.targets)
        sampled_indices = sample_indices(labels, ratio, stratified=stratified, seed=seed)

        # setup data for loaders
        data = Datasets(
            data_name=self.data_name, val_info=sampled_indices,
            train_transform=train_transform,
            val_transform=train_transform,
            test_transform=test_transform
        )
        
        retain_train_loader, forget_train_loader, test_loader = data.get_loader(
            batch_size=batch_size, drop_last_train=drop_last_train, shuffle_train=shuffle_train, shuffle_val=shuffle_train
        )
        
        train_loaders = {
            "Retain": retain_train_loader,
            "Forget": forget_train_loader,
        }
        
        test_loaders = {
            "Test": test_loader,
        }
        
        return train_loaders, test_loaders

from torch.utils.data import DataLoader
import itertools

class MergedLoaders:
    def __init__(self, loaders: dict):
        """
        loaders: {'retain': DataLoader, 'forget': DataLoader, ...} 형식의 딕셔너리
        """
        if not loaders:
            raise ValueError("At least one DataLoader must be provided.")

        self.loaders = loaders
        self.length = max(len(loader) for loader in loaders.values())
        self.reset_iterators()

    def reset_iterators(self):
        """ 각 DataLoader에 대한 iterator를 새로 생성 """
        self.iterators = {}
        for key, loader in self.loaders.items():
            it = iter(loader)
            if len(loader) < self.length:
                self.iterators[key] = itertools.islice(itertools.cycle(it), self.length)
            else:
                self.iterators[key] = it

    def __iter__(self):
        self.reset_iterators()  # 각 epoch마다 새 iterator 생성
        return self

    def __next__(self):
        return {key: next(loader) for key, loader in self.iterators.items()}

    def __len__(self):
        return self.length
