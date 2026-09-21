from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset
import random
import torch

from .utils import create_few_shot_data

class OxfordPets(Dataset):
    def __init__(self, transform_train, transform_test, shot_num=16):

        self.template = "a photo of {}."

        a = OxfordIIITPet(
            root="./data",
            split="trainval",
            download=True,
            transform=transform_train
        )
        b = OxfordIIITPet(
            root="./data",
            split="test",
            download=True,
            transform=transform_test
        )
        self.class_dict = {}
        for idx, name in enumerate(a.classes):
            self.class_dict[idx] = name.lower()

        labels = a._labels

        self.train_set = create_few_shot_data(a, labels, shot_num)

        self.test_set = b

        self.is_train = True

    def train_mode(self):
        self.is_train = True

    def test_mode(self):
        self.is_train = False

    def __len__(self):
        return len(self.train_set) if self.is_train else len(self.test_set)

    def __getitem__(self, index):
        if self.is_train:
            x, y = self.train_set[index]
        else:
            x, y = self.test_set[index]
        return x, y