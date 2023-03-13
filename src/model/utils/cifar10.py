import sys

import torch
import torchvision
import torchvision.transforms as transforms
import utils


class CIFAR10Loader:
    def __init__(self, root: str) -> None:
        self.root = root

    def train(
        self, batch_size: int, num_workers: int, noise_level: int, transform_method: str
    ):

        if transform_method == "flip_crop":
            transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomResizedCrop((32, 32)),
                    transforms.ToTensor(),
                ]
            )
        else:
            sys.exit("存在しない画像変換手法です")

        train_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=True, transform=transform, download=True
        )

        train_dataset = utils.labelnoise.label_noise(train_dataset, noise_level)

        train_loader = torch.utils.data.DataLoader(
            dataset=train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        return train_loader

    def test(self, batch_size: int, num_workers: int):

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
            ]
        )

        test_dataset = torchvision.datasets.CIFAR10(
            root=self.root, train=False, transform=transform, download=True
        )

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        return test_loader
