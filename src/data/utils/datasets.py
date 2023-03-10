import glob
import io
import os
import pickle
import random
import sys

import numpy as np
import torch
import torchvision.transforms as transforms
from natsort import natsorted
from PIL import Image
from tqdm import tqdm


class TestTimeAugCIFAR10:
    base_folder = "CIFAR10"
    test_name = "test_batch"
    aug_data_name = "CIFAR10"
    aug_test_name = "test_batch_aug_10"

    def __init__(self, root: str, transform_name: str) -> None:

        self.root = root
        self.transform_name = transform_name

        file_path = os.path.join(self.root, self.base_folder, self.test_name)
        with open(file_path, "rb") as fo:
            entry = pickle.load(fo, encoding="latin1")

        # now load the picked numpy arrays
        self.data = entry["data"]
        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

        if self.transform_name == "flip_crop":
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                ]
            )
        else:
            sys.exit("存在しない画像変換手法です")

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

    def aug(self, root, aug_num: int):

        output_path = os.path.join(
            root, self.aug_data_name, self.aug_test_name + "_" + self.transform_name
        )
        os.makedirs(os.path.join(root, self.aug_data_name), exist_ok=True)
        origin = []
        for i in tqdm(range(len(self.data)), total=len(self.data)):
            random.seed(2021)
            torch.manual_seed(2021)
            data_pick = Image.fromarray(self.data[i])
            for _ in range(aug_num):
                data_aug = np.array(self.transform(data_pick))
                origin.append(data_aug)
        self.data = np.array(origin)
        self.data = self.data.transpose((0, 3, 1, 2))
        self.data = self.data.reshape(-1, 3072)
        with open(output_path, "wb") as f:
            pickle.dump(self.data, f)


class TestTimeAugSVHN(TestTimeAugCIFAR10):
    base_folder = "SVHN"
    test_name = "test_32x32.mat"
    aug_data_name = "SVHN"
    aug_test_name = "test_batch_aug_10"

    def __init__(self, root: str, transform_name: str) -> None:
        import scipy.io as sio

        self.root = root
        self.transform_name = transform_name

        file_path = os.path.join(self.root, self.base_folder, self.test_name)
        loaded_mat = sio.loadmat(file_path)
        self.data = loaded_mat["X"]
        self.data = np.transpose(self.data, (3, 0, 1, 2))

        pick_num = 10000
        random.seed(2021)
        idx_list = np.array(random.sample(list(range(len(self.data))), pick_num))
        self.data = self.data[idx_list]

        if self.transform_name == "flip_crop":
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                ]
            )
        else:
            sys.exit("存在しない画像変換手法です")


class TestTimeAugLSUN(TestTimeAugCIFAR10):
    base_folder = "LSUN"
    aug_data_name = "LSUN"
    aug_test_name = "test_batch_aug_10"

    def __init__(self, root: str, transform_name: str) -> None:
        import lmdb

        self.root = root
        self.transform_name = transform_name
        self.data = []

        file_path = os.path.join(self.root, self.base_folder)
        env = lmdb.open(
            file_path,
            max_readers=1,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )

        with env.begin(write=False) as txn:
            keys = [key for key in txn.cursor().iternext(keys=True, values=False)]
            for i in range(len(keys)):
                imgbuf = txn.get(keys[i])
                buf = io.BytesIO()
                buf.write(imgbuf)
                buf.seek(0)
                img = np.array(Image.open(buf).convert("RGB"))
                self.data.append(img)
        if self.transform_name == "flip_crop":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                ]
            )
        else:
            sys.exit("存在しない画像変換手法です")


class TestTimeAugTinyImageNet(TestTimeAugCIFAR10):
    base_folder = "ImageNet/images"
    aug_data_name = "ImageNet"
    aug_test_name = "test_batch_aug_10"

    def __init__(self, root: str, transform_name: str) -> None:

        self.root = root
        self.transform_name = transform_name
        self.data = []

        file_path_list = natsorted(
            glob.glob(os.path.join(self.root, self.base_folder, "*"))
        )
        for file_path in file_path_list:
            random.seed(2021)
            torch.manual_seed(2021)
            img = np.array(Image.open(file_path).convert("RGB"))
            self.data.append(img)

        if self.transform_name == "flip_crop":
            self.transform = transforms.Compose(
                [
                    transforms.Resize((32, 32)),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                ]
            )
        else:
            sys.exit("存在しない画像変換手法です")


class TestTimeAugCIFAR10C(TestTimeAugCIFAR10):
    base_folder = "CIFAR10-C"
    aug_data_name = "CIFAR10-C"
    aug_test_name = "test_batch_aug_10"

    def __init__(
        self,
        root: str,
        transform_name: str,
        corruption_name: str,
        corruption_level: int,
    ) -> None:

        self.root = root
        self.transform_name = transform_name
        num_data = 10000

        CORRUPTIONS = [
            "gaussian_noise",
            "shot_noise",
            "impulse_noise",
            "defocus_blur",
            "glass_blur",
            "motion_blur",
            "zoom_blur",
            "snow",
            "frost",
            "fog",
            "brightness",
            "contrast",
            "elastic_transform",
            "pixelate",
            "jpeg_compression",
        ]
        corruption_idx = num_data * corruption_level
        file_path = os.path.join(self.root, self.base_folder, "{}.npy")
        self.data = np.load(file_path.format(corruption_name))[
            corruption_idx - num_data : corruption_idx
        ]

        if self.transform_name == "flip_crop":
            self.transform = transforms.Compose(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(32, padding=4),
                ]
            )
        else:
            sys.exit("存在しない画像変換手法です")
