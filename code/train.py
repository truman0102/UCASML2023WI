import os
import time
import random
import numpy as np

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset

from utils import Cifar10
from model import ResNet50


class EarlyStopping:
    def __init__(self, patience=10, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_loss = 1e15
        self.counter = 0
        self.stop = False

    def save_path(self, path):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
        file = os.path.join(path, current_time + ".pt")
        return file

    def __call__(self, *args, **kwargs):
        pass


def get_data_loader(dataset: Cifar10, batch_size=64, sample_size=1024):
    train_images, train_labels = dataset.get_train()
    test_images, test_labels = dataset.get_test()
    preprocess = transforms.Compose(
        [
            transforms.Resize(256, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    cat = {
        label: np.where(train_labels == label)[0].tolist() for label in range(10)
    }  # 每个类别的图片在训练集中的索引
    if sample_size is None:
        sample_size = 5000
    sample = {
        label: random.sample(cat[label], sample_size) for label in range(10)
    }  # 每个类别随机采样

    X_train_new = np.concatenate([train_images[sample[label]] for label in range(10)])
    y_train_new = np.concatenate([train_labels[sample[label]] for label in range(10)])

    train_dataset = TensorDataset(
        preprocess(torch.from_numpy(X_train_new).float()),
        torch.from_numpy(y_train_new).long(),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )

    test_dataset = TensorDataset(
        preprocess(torch.from_numpy(test_images).float()),
        torch.from_numpy(test_labels).long(),
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )

    return train_loader, test_loader


def get_model(pretrained_path="../model/ResNet/resnet50-19c8e357.pth"):
    model = ResNet50()
    if pretrained_path is not None:
        model.load_state_dict(torch.load(pretrained_path))
    return model


def train(model, train_loader, test_loader, epochs, learning_rate, device, save_path):
    pass


def main(data_path, pretrained_path, save_path):
    dataset = Cifar10(data_path)
    train_loader, test_loader = get_data_loader(dataset)

    model = get_model(pretrained_path)
    model.freeze()  # 冻结卷积层

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
