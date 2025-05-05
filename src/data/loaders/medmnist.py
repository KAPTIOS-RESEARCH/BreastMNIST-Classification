import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from src.data.sets.medmnist import MedMNISTDataset


def create_random_sampler(dataset):
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


class MedMNISTDataloader(object):
    def __init__(self,
                 data_dir: str,
                 dataset_name: str = 'BreastMNIST',
                 batch_size: int = 4,
                 num_workers: int = 4,
                 image_size: int = 128,
                 debug: bool = True):

        super(MedMNISTDataloader, self).__init__()
        self.data_dir = data_dir
        self.dataset_name = dataset_name
        self.debug = debug
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size

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
        train_dataset = MedMNISTDataset(
            self.data_dir,
            self.dataset_name,
            self.image_size,
            split="train"
        )

        if self.debug:
            train_dataset = Subset(
                train_dataset, range(self.batch_size * 2))

        random_sampler = self._create_random_sampler(train_dataset)

        return DataLoader(train_dataset, sampler=random_sampler, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True if torch.cuda.is_available() else False)

    def val(self):
        val_dataset = MedMNISTDataset(
            self.data_dir,
            self.dataset_name,
            self.image_size,
            split="val"
        )
        if self.debug:
            val_dataset = Subset(val_dataset, range(self.batch_size * 2))

        return DataLoader(val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True if torch.cuda.is_available() else False)

    def test(self):
        test_dataset = MedMNISTDataset(
            self.data_dir,
            self.dataset_name,
            self.image_size,
            split="test"
        )
        if self.debug:
            test_dataset = Subset(test_dataset, range(self.batch_size * 2))

        return DataLoader(test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True if torch.cuda.is_available() else False)
