import argparse
import os
import sys

import utils


def main():
    BASE_FOLDER = "dataset"
    LOAD_FOLDER = "external"
    OUTPUT_FOLDER = "processed"
    DATASET_LIST = ["CIFAR10", "CIFAR10-C", "ImageNet", "LSUN", "SVHN"]
    CIFAR10C_CORRUPTION_LIST = [
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

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="", choices=DATASET_LIST)
    parser.add_argument("--transform-method", help="", default="flip_crop")
    parser.add_argument("--augnum", help="", type=int, default=10)
    parser.add_argument("--corruption-level", help="", type=int, default=1)
    parser.add_argument(
        "--corruption-method",
        help="",
        choices=CIFAR10C_CORRUPTION_LIST,
        default="gaussian_noise",
    )
    args = parser.parse_args()

    load_path = os.path.join(".", BASE_FOLDER, LOAD_FOLDER)
    output_path = os.path.join(".", BASE_FOLDER, OUTPUT_FOLDER)
    os.makedirs(output_path, exist_ok=True)

    if args.dataset == "CIFAR10":
        test_dataset = utils.datasets.TestTimeAugCIFAR10(
            root=load_path, transform_name=args.transform_method
        )
        test_dataset.aug(root=output_path, aug_num=args.augnum)

    elif args.dataset == "CIFAR10-C":
        test_dataset = utils.datasets.TestTimeAugCIFAR10C(
            root=load_path,
            transform_name=args.transform_method,
            corruption_name=args.corruption_method,
            corruption_level=args.corruption_level,
        )
        test_dataset.aug(root=output_path, aug_num=10)

    elif args.dataset == "ImageNet":
        test_dataset = utils.datasets.TestTimeAugTinyImageNet(
            root=load_path, transform_name=args.transform_method
        )
        test_dataset.aug(root=output_path, aug_num=args.augnum)

    elif args.dataset == "LSUN":
        test_dataset = utils.datasets.TestTimeAugLSUN(
            root=load_path, transform_name=args.transform_method
        )
        test_dataset.aug(root=output_path, aug_num=args.augnum)

    elif args.dataset == "SVHN":
        test_dataset = utils.datasets.TestTimeAugSVHN(
            root=load_path, transform_name=args.transform_method
        )
        test_dataset.aug(root=output_path, aug_num=args.augnum)
    else:
        sys.exit("存在しないデータセットです")


if __name__ == "__main__":
    main()
