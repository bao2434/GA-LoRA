from torchvision.datasets import Flowers102
from torch.utils.data import Dataset

from .utils import create_few_shot_data


class OxfordFlowers(Dataset):
    def __init__(self, transform_train, transform_test, shot_num=16):

        self.template = "a photo of a {}."

        train_dataset = Flowers102(
            root="./data",
            split="train",
            download=True,
            transform=transform_train
        )

        test_dataset = Flowers102(
            root="./data",
            split="test",
            download=True,
            transform=transform_test
        )

        # Flowers102 does not provide class names directly
        # so we create generic flower class names
        self.class_dict = {}
        for idx, class_name in enumerate(train_dataset.classes):
            self.class_dict[idx] = f"flower {class_name}"

        labels = train_dataset._labels

        self.train_set = create_few_shot_data(
            train_dataset,
            labels,
            shot_num
        )

        self.test_set = test_dataset

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