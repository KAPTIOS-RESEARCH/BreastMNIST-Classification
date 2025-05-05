import wandb
import logging
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
                optimizer=self.optimizer, mode='min', factor=0.1)
        elif lr_scheduler_type == 'exponential':
            self.lr_scheduler = ExponentialLR(
                optimizer=self.optimizer, gamma=0.97)

        # LOSS FUNCTION
        self.criterion = instanciate_module(parameters['loss']['module_name'],
                                            parameters['loss']['class_name'],
                                            parameters['loss']['parameters'])

        self.metric = instanciate_module(parameters['metric']['module_name'],
                                            parameters['metric']['class_name'],
                                            parameters['metric']['parameters'])
        
    def train(self, dl: DataLoader):
        raise NotImplementedError

    def test(self, dl: DataLoader):
        raise NotImplementedError

    def fit(self, train_dl, test_dl, log_dir: str):
        num_epochs = self.parameters['num_epochs']
        for epoch in range(num_epochs):
            train_loss, train_metric = self.train(train_dl)
            test_loss, test_metric = self.test(test_dl)

            if self.parameters['track']:
                wandb.log({
                    f"Train/{self.parameters['loss']['class_name']}": train_loss,
                    f"Test/{self.parameters['loss']['class_name']}": test_loss,
                    f"Train/{self.parameters['metric']['class_name']}": train_metric,
                    f"Test/{self.parameters['metric']['class_name']}": test_metric,
                    "_step_": epoch
                })

            if self.lr_scheduler is not None:
                self.lr_scheduler.step(test_loss)

            logging.info(
                f"Epoch {epoch + 1} / {num_epochs} - Train/Test {self.parameters['loss']['class_name']}: {train_loss:.4f} | {test_loss:.4f} - Train/Test {self.parameters['metric']['class_name']} : {train_metric:.2f} | {test_metric:.2f}")

            if self.early_stop is not None:
                self.early_stop(self.model, test_loss, log_dir, epoch)
                if self.early_stop.stop:
                    logging.info(
                        f"Val loss did not improve for {self.early_stop.patience} epochs.")
                    logging.info(
                        'Training stopped by early stopping mecanism.')
                    break

        if self.parameters['track']:
            wandb.finish()
