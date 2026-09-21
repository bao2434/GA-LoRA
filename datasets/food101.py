from torchvision.datasets import Food101 as food101
from torch.utils.data import Dataset
import random
import torch

from .utils import create_few_shot_data

class Food101(Dataset):
    def __init__(self, transform_train, transform_test, shot_num = 16):

        self.template = "a photo of"

        a = food101(
            root="./data",
            split="train",
            download=False,
            transform=transform_train
        )

        b = food101(
            root="./data",
            split="test",
            download=False,
            transform=transform_test
        )
        
        self.class_dict = {idx: cls for cls, idx in a.class_to_idx.items()}

        self.train_set = create_few_shot_data(a, a._labels, shot_num)
        self.test_set  = b
        self.is_train = True

    def train_mode(self):
        self.is_train = True

    def test_mode(self):
        self.is_train = False

    def __len__(self):
        if (self.is_train):
            return len(self.train_set)
        else:
            return len(self.test_set)

    def __getitem__(self, index):
        if (self.is_train):
            x, y = self.train_set[index]
        else:
            x, y =  self.test_set[index]
        return x, y