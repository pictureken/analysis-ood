import io
import os
import pickle
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets
from tqdm import tqdm


class AugCIFAR10():
    base_folder = "cifar-10-batches-py"
    test_name = "test_batch"
    aug_data_name = "CIFAR10"
    aug_test_name = "test_batch_aug_10"
    def __init__(self, root: str, transform_name: str) -> None:

        self.root = root
        self.transform_name = transform_name
        if transform_name == "flip_crop":
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32,padding=4)
            ])

        file_path = os.path.join(self.root, self.base_folder, self.test_name)
        with open(file_path, 'rb') as fo:
            entry = pickle.load(fo, encoding='latin1')

        # now load the picked numpy arrays
        self.data = entry["data"]
        self.data = np.vstack(self.data).reshape(-1,3,32,32)
        self.data = self.data.transpose((0,2,3,1)) # convert to HWC

    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img = self.data[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        return img
    
    def __len__(self) -> int:
        return len(self.data)
    
    def image_aug(self, root, aug_num: int):
        origin = []
        for i in tqdm(range(10000),total = 10000):
            random.seed(2021)
            torch.manual_seed(2021)
            data_pick = Image.fromarray(self.data[i])
            for _ in range(aug_num):
                data_aug = np.array(self.transform(data_pick))
                origin.append(data_aug)
        origin = np.array(origin)
        origin = origin.transpose((0,3,1,2))
        origin = origin.reshape(-1,3072)
        self.data = origin
        # with open(root, "wb") as f:
        #     pickle.dump(self.data, f)
 

if __name__ == "__main__":
    aug = AugCIFAR10(root="./dataset/external",transform_name="flip_crop")
    print(len(aug))
    aug.image_aug(root="",aug_num=10)
    print(len(aug))