import pytorch_lightning as pl
import inspect
import importlib
import torch
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import INbreastDataset
from losses import f1_precis_recall


class INBreastClassification(pl.LightningModule):
    def __init__(self, config):
        super(INBreastClassification, self).__init__()
        self.config = config
        self.n_class = len(config['data']['groups'])
        self.criteria = nn.CrossEntropyLoss()
        self.epoch_outputs = list()
        self.epoch_true_labels = list()
        self.val_epoch_outputs = list()
        self.val_epoch_true_labels = list()
        self.__load_model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'train', self.epoch_true_labels, self.epoch_outputs)

    def validation_step(self, batch, batch_idx):
        return self._shared_eval(batch, batch_idx, 'val', self.val_epoch_true_labels, self.val_epoch_outputs)

    def _shared_eval(self, batch, batch_idx, prefix, true_labels: list, pred: list):
        result_string = 'loss' if prefix == 'train' else f'{prefix}_loss'
        x, y = batch
        y_pred = self.forward(x)
        pred.append(y_pred)
        true_labels.append(y)
        loss = self.criteria(y_pred, y)
        loss = loss.unsqueeze(dim=-1)
        self.logger.experiment.add_scalar(f'Loss/{prefix}', loss, self.global_step)
        return {result_string: loss}

    def on_epoch_end(self):
        return self.calculate_metrics('train', self.epoch_true_labels, self.epoch_outputs)

    def validation_epoch_end(self, outputs: list):
        return self.calculate_metrics('val', self.val_epoch_true_labels, self.val_epoch_outputs)

    def calculate_metrics(self, prefix: str, targets: list, pred: list,):
        outputs = torch.cat(pred, dim=0)
        true_labels = torch.cat(targets, dim=0)
        _, predicted = torch.max(outputs.data, 1)
        total = len(true_labels)
        correct = (predicted == true_labels).sum()
        accuracy = 100 * correct / total
        scores = f1_precis_recall(outputs, true_labels)

        pred.clear()
        targets.clear()

        self.logger.experiment.add_scalar(f"F1/{prefix}", scores['F1'], self.current_epoch)
        self.logger.experiment.add_scalar(f"Accuracy/{prefix}", accuracy, self.current_epoch)
        self.logger.experiment.add_scalar(f"Precision/{prefix}", scores['precision'], self.current_epoch)
        self.logger.experiment.add_scalar(f"Recall/{prefix}", scores['recall'], self.current_epoch)
        results_dict = {f"F1/{prefix}": scores['F1'], f"Accuracy/{prefix}": accuracy,
                        f"Precision/{prefix}": scores['precision'], f"Recall/{prefix}": scores['recall']}
        return results_dict

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        root_dir = self.config["data"]["data_dir"]
        val_coefficient = float(self.config["data"]["val_coefficient"])
        dataset = eval(self.config["data"]["data_set"])
        train_dataset = dataset(root_dir, partition="train", config=self.config["data"], transform=transform)
        train_dataset, val_dataset = random_split(train_dataset, [int(len(train_dataset) * (1.0 - val_coefficient)),
                                                                  int(len(train_dataset) * val_coefficient)])
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=16, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=16, shuffle=True)

    def __load_model(self):
        mapping = self.__module_mapping('models')
        if 'parameters' not in self.config['model']:
            self.config['model']['parameters'] = {}
        self.config['model']['parameters']['n_classes'] = self.n_class
        self.model = mapping[self.config['model']['name']](**self.config['model']['parameters']).get_model()

    @staticmethod
    def __module_mapping(module_name):
        mapping = {}
        for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
            mapping[name] = obj
        return mapping
