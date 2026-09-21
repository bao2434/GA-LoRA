from torchvision.datasets import Caltech101
from torch.utils.data import Dataset
from torch.utils.data import random_split
import torch

from .utils import create_few_shot_data


class Caltech101Dataset(Dataset):
    def __init__(
        self,
        transform_train,
        transform_test,
        shot_num=16,
        train_ratio=0.8,
        seed=1
    ):

        self.template = "a photo of a {}."

        full_dataset = Caltech101(
            root="./data",
            download=True,
            transform=transform_train
        )

        # Build class dictionary
        self.class_dict = {}
        for idx, name in enumerate(full_dataset.categories):
            self.class_dict[idx] = name.replace("_", " ").lower()

        # Create train/test split manually
        total_size = len(full_dataset)
        train_size = int(train_ratio * total_size)
        test_size = total_size - train_size

        generator = torch.Generator().manual_seed(seed)

        train_subset, test_subset = random_split(
            full_dataset,
            [train_size, test_size],
            generator=generator
        )

        # Extract labels for few-shot sampling
        labels = [
            full_dataset.y[idx]
            for idx in train_subset.indices
        ]

        # Few-shot train set
        self.train_set = create_few_shot_data(
            train_subset,
            labels,
            shot_num
        )

        # Rebuild test dataset with test transform
        full_dataset_test = Caltech101(
            root="./data",
            download=True,
            transform=transform_test
        )

        self.test_set = torch.utils.data.Subset(
            full_dataset_test,
            test_subset.indices
        )

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