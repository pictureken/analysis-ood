import argparse
import os

import utils


def main():
    BASE_FOLDER = "dataset"
    LOAD_FOLDER = "raw"
    NOISE_LEVEL_LIST = [0, 5, 10, 15, 20]
    MODEL_LIST = ["ResNet18"]
    TRANSFORM_LIST = ["flip_crop"]
    CIFAR10_CLASSES = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "track",
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-batch-size", help="", type=int, default=128)
    parser.add_argument("--test-batch-size", help="", type=int, default=100)
    parser.add_argument("--num-workers", help="", type=int, default=2)
    parser.add_argument("--epoch", help="", type=int, default=4000)
    parser.add_argument("--learning-late", help="", type=float, default=1e-4)
    parser.add_argument("--gpu-device", help="", type=str, default="cuda:0")
    parser.add_argument("--model", help="", choices=MODEL_LIST, default="ResNet18")
    parser.add_argument("--model-size", help="", type=int, default=64)
    parser.add_argument(
        "--noise-level", help="", type=int, choices=NOISE_LEVEL_LIST, default=0
    )
    parser.add_argument(
        "--transform-method", help="", choices=TRANSFORM_LIST, default="flip_crop"
    )
    args = parser.parse_args()

    # dataloader
    load_path = os.path.join(".", BASE_FOLDER, LOAD_FOLDER)
    loader = utils.data.CIFAR10Loader(root=load_path)
    train_loader = loader.train(
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        noise_level=args.noise_level,
        transform_method=args.transform_method,
    )
    test_loader = loader.test(
        batch_size=args.test_batch_size, num_workers=args.num_workers
    )

    trainer = utils.trainer.TrainModel(
        lr=args.learning_late,
        gpu=args.gpu_device,
        model_name=args.model,
        model_size=args.model_size,
        num_classes=len(CIFAR10_CLASSES),
    )

    for i in range(args.epoch):
        trainer.train(train_loader)
        trainer.eval(test_loader)


if __name__ == "__main__":
    main()
