from __future__ import print_function, division

import os
import pandas as pd
import cv2
import yaml
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class INbreastDataset(Dataset):
    LABELS_FILENAME = "INbreast_pre.csv"
    IMG_PATH_COLUMN = 'IMG_PATH'
    IMAGE_FOLDER = "data"
    LABELS_NAME = "Bi-Rads"

    def __init__(self, root, partition='train', transform=None, augmentation=None, config=dict()):
        self.root_dir = root
        self.partition = partition
        self.config = config
        self.__load_data()
        self.__find_labels()
        self.transform = transform
        self.augmentation = augmentation

    def __load_data(self):
        self.df = pd.read_csv(os.path.join(self.root_dir, self.partition, self.LABELS_FILENAME))

    def __find_labels(self):
        mapper = {}

        for group, values in self.config['groups'].items():
            for value in values:
                mapper[value] = group

        self.labels = self.df[self.LABELS_NAME].apply(lambda x: mapper[x])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        info = self.df.iloc[idx]
        path = os.path.join(self.root_dir, self.partition, self.IMAGE_FOLDER, info[self.IMG_PATH_COLUMN])

        y = self.labels[idx]
        X = cv2.imread(path)

        if self.augmentation:
            X = self.augmentation(X)

        if self.transform:
            X = self.transform(X)

        return X, y


if __name__ == "__main__":
    root_dir = '/Users/yuriiyelisieiev/Desktop/Machine_Learning/Mammgraphy'

    with open('./config.yaml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = INbreastDataset(root_dir, config=config['data'], transforms=transform)

    train_loader = DataLoader(dataset, batch_size=16)
    print(len(train_loader))

    for img, label in train_loader:
        print(img)
