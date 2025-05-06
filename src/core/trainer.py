import wandb
import logging
import time
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau
from src.utils.config import instanciate_module
from src.optimisation.early_stopping import EarlyStopping
from src.models.base import BaseTorchModel

class BaseTrainer(object):

    def __init__(self, model: BaseTorchModel, parameters: dict, device: str):
        self.model = model
        self.parameters = parameters
        self.device = device
        self.early_stop = EarlyStopping(
            patience=parameters['early_stopping_patience'], enable_wandb=parameters['track']) if parameters['early_stopping_patience'] else None

        self.optimizer = Adam(
            self.model.parameters(),
            lr=parameters['lr'],
            weight_decay=parameters['weight_decay']
        )

        self.lr_scheduler = None
        lr_scheduler_type = parameters['lr_scheduler'] if 'lr_scheduler' in parameters.keys(
        ) else 'none'

        if lr_scheduler_type == 'cosine':
            self.lr_scheduler = CosineAnnealingLR(
                optimizer=self.optimizer, T_max=100)
        elif lr_scheduler_type == 'plateau':
            self.lr_scheduler = ReduceLROnPlateau(
                optimizer=self.optimizer, mode='min', factor=0.1, patience = 2)
        elif lr_scheduler_type == 'exponential':
            self.lr_scheduler = ExponentialLR(
                optimizer=self.optimizer, gamma=0.97)

        self.is_binary = getattr(self.model, 'n_output', 1) == 1
        
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                    parameters['loss']['class_name'],
                                    parameters['loss']['parameters'])
        
        if not self.criterion:
            if self.is_binary:
                self.criterion = nn.BCELoss()
            else:
                self.criterion = nn.MSELoss()

        self.metrics = {}
        
        for metric in parameters['metrics']:
            metric_instance = instanciate_module(
                metric['module_name'], metric['class_name'], metric['parameters'])
            self.metrics[metric['class_name']] = metric_instance
        
    def train(self, dl: DataLoader):
        raise NotImplementedError

    def test(self, dl: DataLoader):
        raise NotImplementedError

    def _format_targets(self, targets):
        if self.is_binary:
            return targets.float().unsqueeze(1)
        else:
            return nn.functional.one_hot(targets.long(), num_classes=self.model.n_output).float()

    def get_metrics(self, all_preds, all_targets):
        if self.metrics is None:
            return None
        
        preprocess_map = {
            'Accuracy': lambda preds, targets: 
                (preds.squeeze() if preds.shape[1] == 1 else preds.argmax(dim=1), targets),
            'AUROC': lambda preds, targets: 
                (preds.squeeze() if preds.shape[1] == 1 else preds.softmax(dim=1), targets),
        }

        metric_results = {}
        for key, metric in self.metrics.items():
            if key in preprocess_map:
                preds, targets = preprocess_map[key](all_preds, all_targets)
                metric_results[key] = metric(preds, targets)
            else:
                metric_results[key] = metric(all_preds, all_targets)

        return metric_results
        
    def fit(self, train_dl, test_dl, log_dir: str):
        start_time = time.time()
        num_epochs = self.parameters['num_epochs']
        
        for epoch in range(num_epochs):
            train_loss, train_metrics = self.train(train_dl)
            test_loss, test_metrics = self.test(test_dl)

            if self.parameters['track']:
                log_data = {
                    f"Train/{self.parameters['loss']['class_name']}": train_loss,
                    f"Test/{self.parameters['loss']['class_name']}": test_loss,
                    "_step_": epoch
                }
                if train_metrics:
                    for metric_name, value in train_metrics.items():
                        log_data[f"Train/{metric_name}"] = value
                if test_metrics:
                    for metric_name, value in test_metrics.items():
                        log_data[f"Test/{metric_name}"] = value

                wandb.log(log_data)

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(test_loss)

            
            metric_log_str = " | ".join([
                f"{name}: {train_metrics[name]:.4f} / {test_metrics[name]:.4f}"
                for name in train_metrics
            ]) if train_metrics else None
            
            logging.info(
                f"Epoch {epoch + 1} / {num_epochs} - "
                f"Loss: {train_loss:.4f} / {test_loss:.4f} - "
                f"Metrics: {metric_log_str}"
            )

            if self.early_stop is not None:
                self.early_stop(self.model, test_loss, log_dir, epoch)
                if self.early_stop.stop:
                    logging.info(
                        f"Val loss did not improve for {self.early_stop.patience} epochs.")
                    logging.info(
                        'Training stopped by early stopping mechanism.')
                    break

        end_time = time.time()
        logging.info(f"Training completed in {end_time - start_time:.2f} seconds.")
        
        if self.parameters['track']:
            wandb.finish()