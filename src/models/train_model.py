import argparse
import os

import torch
import torchvision
import torchvision.transforms as transforms
import utils


def main():
    BASE_FOLDER = "dataset"
    LOAD_FOLDER = "raw"
    MODEL_LIST = ["ResNet18"]

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-batch-size", help="", type=int, default=128)
    parser.add_argument("--test-batch-size", help="", type=int, default=100)
    parser.add_argument("--num-workers", help="", type=int, default=2)
    parser.add_argument("--noise-level", help="", type=int, default=20)
    parser.add_argument("--model", help="", choices=MODEL_LIST)
    args = parser.parse_args()

    # dataloader
    load_path = os.path.join(".", BASE_FOLDER, LOAD_FOLDER)
    loader = utils.data.CIFAR10Loader(root=load_path)
    train_loader = loader.train(
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        noise_level=args.noise_level,
    )
    test_loader = loader.test(
        batch_size=args.test_batch_size, num_workers=args.num_workers
    )


if __name__ == "__main__":
    main()
