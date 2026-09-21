from torchvision.datasets import EuroSAT
from torch.utils.data import Dataset
import torch

from .utils import create_few_shot_data

NEW_CNAMES = {
    'AnnualCrop': 'Annual Crop Land',
    'Forest': 'Forest',
    'HerbaceousVegetation': 'Herbaceous Vegetation Land',
    'Highway': 'Highway or Road',
    'Industrial': 'Industrial Buildings',
    'Pasture': 'Pasture Land',
    'PermanentCrop': 'Permanent Crop Land',
    'Residential': 'Residential Buildings',
    'River': 'River',
    'SeaLake': 'Sea or Lake'
}


class EuroSatDataset(Dataset):
    def __init__(self, transform_train, transform_test, shot_num=16):

        self.template = "a photo of"

        # torchvision EuroSAT
        full_dataset = EuroSAT(
            root="./data",
            download=True
        )

        # Create train/test split manually
        dataset_size = len(full_dataset)
        train_size = int(0.8 * dataset_size)
        test_size = dataset_size - train_size

        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset,
            [train_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )

        self.transform_train = transform_train
        self.transform_test = transform_test

        # Build class name dictionary
        self.class_dict = {
            idx: NEW_CNAMES[name]
            for idx, name in enumerate(full_dataset.classes)
        }

        # Extract labels for few-shot creation
        train_targets = [
            full_dataset.targets[idx]
            for idx in train_dataset.indices
        ]

        self.train_set = create_few_shot_data(
            train_dataset,
            train_targets,
            shot_num
        )

        self.test_set = test_dataset

        self.is_train = True

    def train_mode(self):
        self.is_train = True

    def test_mode(self):
        self.is_train = False

    def __len__(self):
        if self.is_train:
            return len(self.train_set)
        else:
            return len(self.test_set)

    def __getitem__(self, index):

        if self.is_train:
            sample = self.train_set[index]

            # Depending on your create_few_shot_data format
            if isinstance(sample, dict):
                img = sample["image"]
                label = sample["label"]
            else:
                img, label = sample

            if self.transform_train:
                img = self.transform_train(img)

        else:
            img, label = self.test_set[index]

            if self.transform_test:
                img = self.transform_test(img)

        return img, label