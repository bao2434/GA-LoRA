from torchvision.datasets import StanfordCars as cars
from torch.utils.data import Dataset
from torchvision import transforms
import random
import torch

from .utils import create_few_shot_data

class StanfordCars(Dataset):
    def __init__(self,  transform_train, transform_test, shot_num = 16):

        self.template = "a photo of"
        data_root = "/kaggle/input/datasets/icebearthecoldcoder/stanford-cars"

        a = cars(
            root=data_root,
            split="train",
            download=False,
            transform=transform_train      # augmented for training
        )

        b = cars(
            root=data_root,
            split="test",
            download=False,
            transform=transform_test       # deterministic for test
        )

        self.class_dict = {}
        for cls, idx in a.class_to_idx.items():
            names = cls.split(' ')
            year = names.pop(-1)
            names.insert(0, year)
            classname = ' '.join(names)
            self.class_dict.setdefault(idx, classname)

        self.train_set = create_few_shot_data(a, [label for _,label in a._samples], shot_num)
        self.test_set  = b
        self.is_train  = True

    def train_mode(self):
        self.is_train = True

    def test_mode(self):
        self.is_train = False

    def __len__(self):
        return len(self.train_set) if self.is_train else len(self.test_set)

    def __getitem__(self, index):
        if self.is_train:
            return self.train_set[index]
        return self.test_set[index]