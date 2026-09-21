from torch.utils.data import Subset
import random


def create_few_shot_data(data, label_list, shot_num = 16):
    class_indices = {}

    for idx, label in enumerate(label_list):
        class_indices.setdefault(label, []).append(idx)

    train_indices = []

    for label, indices in class_indices.items():
        random.shuffle(indices)
        train_indices.extend(indices[:shot_num])

    return Subset(data, train_indices)

