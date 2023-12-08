import cv2
import os
import pickle
import numpy as np


def unpickle(file):
    with open(file, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    return dict


class Cifar10:
    """
    A class to load the cifar10 dataset.
    """

    def __init__(self, path):
        self.path = path  # path to the cifar10 dataset
        meta_file = [file for file in os.listdir(path) if file.__contains__("meta")][0]
        self.meta = unpickle(os.path.join(path, meta_file))
        train_files = [
            file for file in os.listdir(path) if file.__contains__("data_batch")
        ]
        self.train = [unpickle(os.path.join(path, file)) for file in train_files]
        test_file = [
            file for file in os.listdir(path) if file.__contains__("test_batch")
        ][0]
        self.test = unpickle(os.path.join(path, test_file))
        self.label_names = [name.decode("utf-8") for name in self.meta[b"label_names"]]

    def convert_to_images(self, batch):
        """
        A batch is a 10000 * 3072 numpy array of uint8s.
        Return a 10000 * 32 * 32 * 3 numpy array of uint8s.
        """
        red_channel = batch[:, :1024].reshape(-1, 32, 32)
        green_channel = batch[:, 1024:2048].reshape(-1, 32, 32)
        blue_channel = batch[:, 2048:].reshape(-1, 32, 32)
        images = np.stack([red_channel, green_channel, blue_channel], axis=3)
        return images  # (B, H, W, C)

    def transform(self, batch):
        """
        Transform a batch of images to a batch of vectors.
        """
        batch = batch / 255.0
        batch = np.transpose(batch, (0, 3, 1, 2))  # (B, C, H, W)
        return batch

    def get_train(self, flatten=False):
        if flatten:
            train_images = np.concatenate([batch[b"data"] for batch in self.train])
        else:
            train_images = self.transform(
                np.concatenate(
                    [self.convert_to_images(batch[b"data"]) for batch in self.train]
                )
            )
        train_labels = np.concatenate([batch[b"labels"] for batch in self.train])
        return train_images, train_labels

    def get_test(self, flatten=False):
        if flatten:
            test_images = self.test[b"data"]
        else:
            test_images = self.transform(self.convert_to_images(self.test[b"data"]))
        test_labels = np.array(self.test[b"labels"])
        return test_images, test_labels

    def label_to_name(self, label):
        return self.label_names[label]


if __name__ == "__main__":
    path = "../data/cifar-10-batches-py"
    dataset = Cifar10(path)
    test_images, test_labels = dataset.get_test()
    print(test_images.shape)
