import torch
from torch import nn
from tqdm import tqdm
from src.core.trainer import BaseTrainer
from torchmetrics import Accuracy
from src.models.base import BaseTorchModel

class CNNTrainer(BaseTrainer):
    def __init__(self, model: BaseTorchModel, parameters: dict, device: str):
        super(CNNTrainer, self).__init__(model, parameters, device)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0.0
        all_preds = []
        all_targets = []
        with tqdm(train_loader, leave=False, desc="Running training phase") as pbar:
            for sample in train_loader:
                data, targets = sample
                data, targets = data.to(self.device), targets.to(self.device)
                formatted_targets = self._format_targets(targets)
                self.optimizer.zero_grad()
                outputs = self.model(data)
                loss = self.criterion(outputs, formatted_targets)
                loss.backward()
                self.optimizer.step()
                train_loss += loss.item()
                all_preds.append(outputs)
                all_targets.append(targets)
                pbar.update(1)

        all_targets = torch.cat(all_targets).cpu()
        all_preds = torch.cat(all_preds).cpu()
        
        train_metrics = self.get_metrics(all_preds, all_targets)
        train_loss /= len(train_loader)
        return train_loss, train_metrics

    def test(self, val_loader):
        self.model.eval()
        test_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            with tqdm(val_loader, leave=False, desc="Running testing phase") as pbar:
                for idx, sample in enumerate(val_loader):
                    data, targets = sample
                    data, targets = data.to(self.device), targets.to(self.device)
                    formatted_targets = self._format_targets(targets)
                    outputs = self.model(data)
                    loss = self.criterion(outputs, formatted_targets)
                    test_loss += loss.item()
                    all_preds.append(outputs)
                    all_targets.append(targets)
                    pbar.update(1)

        all_targets = torch.cat(all_targets).cpu()
        all_preds = torch.cat(all_preds).cpu()
        test_metrics = self.get_metrics(all_preds, all_targets)
        test_loss /= len(val_loader)
        return test_loss, test_metrics
