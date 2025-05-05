from torch.utils.data import Subset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor, Normalize, Compose, Grayscale
from torchvision.datasets import FashionMNIST, CIFAR10
import numpy as np
import torch

class DefaultDataloader(object):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 debug: bool = True):

        super(DefaultDataloader, self).__init__()
        self.data_dir = data_dir
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = Compose([
            ToTensor(),
        ])

    def train(self):
        train_dataset = FashionMNIST(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True
        )

        if self.debug:
            train_dataset = Subset(train_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=True,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = FashionMNIST(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True
        )

        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))

        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()


class CIFAR10DataLoader(object):
    def __init__(self,
                 data_dir: str,
                 batch_size: int = 4,
                 num_workers: int = 4,
                 debug: bool = False):

        super(CIFAR10DataLoader, self).__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.debug = debug

        self.transform = Compose([
            Grayscale(),
            ToTensor(),
            Normalize((0,), (1,))
        ])
    
    def _create_random_sampler(self, dataset):
        labels = np.array([label for _, label in dataset])
        if labels.ndim > 1:
            labels = labels.flatten()
        labels = torch.tensor(labels, dtype=torch.int64)
        class_counts = torch.bincount(labels)
        class_weights = 1.0 / class_counts.float()
        sampler_weights = [class_weights[label] for _, label in dataset]
        sampler = WeightedRandomSampler(
            weights=sampler_weights, num_samples=len(dataset), replacement=True)
        return sampler

    def train(self):
        train_dataset = CIFAR10(
            root=self.data_dir,
            train=True,
            transform=self.transform,
            download=True
        )

        random_sampler = self._create_random_sampler(train_dataset)
        
        if self.debug:
            train_dataset = Subset(train_dataset, range(self.batch_size * 2))
            
        dataloader = DataLoader(train_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                sampler=random_sampler,
                                pin_memory=True)
        return dataloader

    def val(self):
        val_dataset = CIFAR10(
            root=self.data_dir,
            train=False,
            transform=self.transform,
            download=True
        )

        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))
            
        dataloader = DataLoader(val_dataset,
                                batch_size=self.batch_size,
                                num_workers=self.num_workers,
                                shuffle=False,
                                pin_memory=True)
        return dataloader

    def test(self):
        return self.val()