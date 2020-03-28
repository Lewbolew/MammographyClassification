import pytorch_lightning as pl
import inspect
import importlib
import torch
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
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
        self.train_epoch_outputs = list()
        self.train_epoch_true_labels = list()
        self.__load_model()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.forward(x)
        self.train_epoch_outputs.append(y_pred)
        self.train_epoch_true_labels.append(y)
        loss = self.criteria(y_pred, y)
        loss = loss.unsqueeze(dim=-1)
        self.logger.experiment.add_scalar('Loss/train', loss, self.global_step)
        return {'loss': loss}

    def on_epoch_end(self):
        outputs = torch.cat(self.train_epoch_outputs, dim=0)
        true_labels = torch.cat(self.train_epoch_true_labels, dim=0)
        _, predicted = torch.max(outputs.data, 1)
        total = len(true_labels)
        correct = (predicted == true_labels).sum()
        accuracy = 100 * correct / total
        scores = f1_precis_recall(outputs, true_labels)
        self.train_epoch_outputs.clear()
        self.train_epoch_true_labels.clear()

        self.logger.experiment.add_scalar("F1/train", scores['F1'], self.current_epoch)
        self.logger.experiment.add_scalar("Accuracy/train", accuracy, self.current_epoch)
        self.logger.experiment.add_scalar("Precision/train", scores['precision'], self.current_epoch)
        self.logger.experiment.add_scalar("Recall/train", scores['recall'], self.current_epoch)

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def train_dataloader(self):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        root_dir = self.config["data"]["data_dir"]
        dataset = eval(self.config["data"]["data_set"])
        train_dataset = dataset(root_dir, partition="train", config=self.config["data"], transform=transform)

        return DataLoader(train_dataset, batch_size=16, shuffle=True)

    # TODO: Implement validation dataloader

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
