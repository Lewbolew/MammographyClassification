import torch
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torchvision import models


class ResNet50:
    def __init__(self, n_classes, pretrained=False):
        self.model = models.resnet50(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def get_model(self):
        return self.model