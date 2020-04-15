import torch
import torchvision
from torch.nn import functional as F
from torch import nn
import pytorch_lightning as pl
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
import cv2


class ResNet50:
    def __init__(self, n_classes, pretrained=False):
        self.model = models.resnet50(pretrained=pretrained)
        features_num = self.model.fc.in_features
        self.model.fc = nn.Linear(features_num, n_classes)

    def get_model(self):
        return self.model


class DetectorNET(nn.Module):
    # DM Challenge Yaroslav Nikulin (Therapixel)
    def __init__(self, num_classes):
        super(DetectorNET, self).__init__()
        self.n_classes = num_classes
        self.conv_net = nn.Sequential(
            nn.BatchNorm2d(3, momentum=0.9, affine=True),
            nn.Conv2d(3, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32, momentum=0.9, affine=True),
            nn.Conv2d(32, 32, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(32, momentum=0.9, affine=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64, momentum=0.9, affine=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(64, momentum=0.9, affine=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128, momentum=0.9, affine=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(128, momentum=0.9, affine=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(256, momentum=0.9, affine=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.BatchNorm2d(256, momentum=0.9, affine=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.9, affine=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.fc = nn.Sequential(
            nn.BatchNorm2d(512, momentum=0.9, affine=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(1024, momentum=0.9, affine=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(512, momentum=0.9, affine=True),
            nn.Conv2d(512, self.n_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        print(x.shape)
        x = self.conv_net(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        print(x.shape)
        return x


if __name__ == "__main__":
    img = torch.rand((1, 3, 224, 244))
    # img = torch.from_numpy(cv2.imread('/Users/yuriiyelisieiev/Desktop/Machine_Learning/Mammgraphy/train/data/53587014_809e3f43339f93c6_MG_L_CC_ANON.png'))
    net = DetectorNET(10)
    net.forward(img.float())
