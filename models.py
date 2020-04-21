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


class VGGBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VGGBlock, self).__init__()
        self.model_block = nn.Sequential(
            nn.BatchNorm2d(in_channels, momentum=0.9, affine=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(out_channels, momentum=0.9, affine=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, x):
        return self.model_block(x)


class DetectorNET(nn.Module):
    # DM Challenge Yaroslav Nikulin (Therapixel)
    def __init__(self, n_classes):
        super(DetectorNET, self).__init__()
        self.n_classes = n_classes
        self.conv_net = nn.Sequential(*[VGGBlock(i[0], i[1]) for i in [(3, 32), (32, 64), (64, 128), (128, 256), (256, 512)]])

        self.fc = nn.Sequential(
            nn.BatchNorm2d(512, momentum=0.9, affine=True),
            nn.Conv2d(512, 1024, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            # nn.BatchNorm2d(1024, momentum=0.9, affine=True),
            nn.Conv2d(1024, 512, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            # nn.BatchNorm2d(512, momentum=0.9, affine=True),
            nn.Conv2d(512, self.n_classes, kernel_size=1, stride=1)
        )

    def forward(self, x):
        x = self.conv_net(x)
        x = self.fc(x).squeeze()
        return x


if __name__ == "__main__":
    img = torch.rand((1, 3, 548, 548))
    # img = torch.from_numpy(cv2.imread('/Users/yuriiyelisieiev/Desktop/Machine_Learning/Mammgraphy/train/data/53587014_809e3f43339f93c6_MG_L_CC_ANON.png'))
    net = DetectorNET(5)
    print(net.forward(img.float()))