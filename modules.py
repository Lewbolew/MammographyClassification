import matplotlib

matplotlib.use('Agg')
import pytorch_lightning as pl
import inspect
import importlib
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from torch.optim import Adam, lr_scheduler
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
from torchvision import transforms
from pytorch_lightning.callbacks import ModelCheckpoint
from datasets import INbreastDataset, INBreastPatchesDataset
from losses import f1_precis_recall
from samplers import ImbalancedDatasetSampler
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
import pandas as pd

torch.manual_seed(0)


class INBreastClassification(pl.LightningModule):
    def __init__(self, config):
        super(INBreastClassification, self).__init__()
        self.config = config
        self.n_class = len(config['data']['groups'])
        self.criteria = nn.CrossEntropyLoss()  # (weight=torch.Tensor([0.0004, 0.0003, 0.0004])
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

    def _shared_eval(self, batch, batch_idx, prefix, true_labels, pred):
        result_string = 'loss' if prefix == 'train' else f'{prefix}_loss'
        x, y = batch
        y_pred = self.forward(x)
        pred.append(y_pred)
        true_labels.append(y)
        loss = self.criteria(y_pred, y)
        loss = loss.unsqueeze(dim=-1)
        if prefix != 'val':
            self.logger.experiment.add_scalar(f'Loss/{prefix}', loss, self.global_step)
        return {result_string: loss}

    def on_epoch_end(self):
        return self.calculate_metrics('train', self.epoch_true_labels, self.epoch_outputs)

    def validation_epoch_end(self, outputs):
        return self.calculate_metrics('val', self.val_epoch_true_labels, self.val_epoch_outputs)

    def calculate_metrics(self, prefix, targets, pred):
        outputs = torch.cat(pred, dim=0)
        true_labels = torch.cat(targets, dim=0)
        _, predicted = torch.max(outputs.data, 1)
        conf_matrix = confusion_matrix(true_labels.cpu().numpy(), predicted.cpu().numpy())
        total = len(true_labels)
        correct = (predicted == true_labels).sum()
        accuracy = 100 * correct / total
        scores = f1_precis_recall(outputs, true_labels, self.n_class)

        pred.clear()
        targets.clear()

        val_los = None
        if prefix == 'val':
            val_los = self.criteria(outputs, true_labels)
            self.logger.experiment.add_scalar("Loss/val", val_los, self.current_epoch)

        fig = plt.figure()
        df = pd.DataFrame(conf_matrix, index=range(5), columns=range(5))
        ax = sns.heatmap(df, annot=True, cmap="coolwarm", fmt='.2f')
        ax.set(xlabel='Predicted label', ylabel='True label')
        data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        self.logger.experiment.add_image(f"confusionMatrix/{prefix}", data, dataformats="HWC")

        self.logger.experiment.add_scalar(f"F1/{prefix}", scores['F1'], self.current_epoch)
        self.logger.experiment.add_scalar(f"Accuracy/{prefix}", accuracy, self.current_epoch)
        self.logger.experiment.add_scalar(f"Precision/{prefix}", scores['precision'], self.current_epoch)
        self.logger.experiment.add_scalar(f"Recall/{prefix}", scores['recall'], self.current_epoch)
        results_dict = {f"F1/{prefix}": scores['F1'], f"Accuracy/{prefix}": accuracy,
                        f"Precision/{prefix}": scores['precision'], f"Recall/{prefix}": scores['recall'],
                        f"{prefix}_loss": val_los}
        return results_dict

    def configure_optimizers(self):
        optimizer = Adam(self.model.parameters(), lr=0.001)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
        return [optimizer], [scheduler]

    def prepare_data(self):
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            # transforms.RandomRotation(20),
            transforms.RandomAffine(20, scale=(0.8, 1.2), shear=(-0.2, 0.2)),
            transforms.ToTensor()
        ])
        root_dir = self.config["data"]["data_dir"]
        val_coefficient = float(self.config["data"]["val_coefficient"])
        dataset = eval(self.config["data"]["data_set"])
        train_dataset = dataset(root_dir, config=self.config["data"], transform=transform)
        train_dataset, val_dataset = random_split(train_dataset, [round(len(train_dataset) * (1.0 - val_coefficient)),
                                                                  round(len(train_dataset) * val_coefficient)])
        self.train_dataset = train_dataset.dataset
        self.val_dataset = val_dataset.dataset

        images, labels = next(iter(self.train_dataloader(16)))

        grid = torchvision.utils.make_grid(images)
        self.logger.experiment.add_image("images", grid)

    def train_dataloader(self, batch_size=128):
        return DataLoader(self.train_dataset, batch_size=batch_size, shuffle=False,
                          sampler=ImbalancedDatasetSampler(self.train_dataset), num_workers=14)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=128, shuffle=False, num_workers=14)

    def __load_model(self):
        mapping = self.__module_mapping('models')
        if 'parameters' not in self.config['model']:
            self.config['model']['parameters'] = {}
        self.config['model']['parameters']['n_classes'] = self.n_class
        self.model = mapping[self.config['model']['name']](**self.config['model']['parameters'])

    @staticmethod
    def __module_mapping(module_name):
        mapping = {}
        for name, obj in inspect.getmembers(importlib.import_module(module_name), inspect.isclass):
            mapping[name] = obj
        return mapping
