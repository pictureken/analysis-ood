import argparse
import datetime
import os

import torch
import utils


def main():
    BASE_FOLDER = "dataset"
    LOAD_FOLDER = "raw"
    MODEL_BASE_FOLDER = "models"
    CSV_BASE_FOLDER = "logs"
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
    loader = utils.cifar10.CIFAR10Loader(root=load_path)
    train_loader = loader.train(
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        noise_level=args.noise_level,
        transform_method=args.transform_method,
    )
    test_loader = loader.test(
        batch_size=args.test_batch_size, num_workers=args.num_workers
    )

    model_setup_path = os.path.join(
        args.model,
        "labelnoise" + str(args.noise_level),
        args.transform_method,
        "modelsize" + str(args.model_size),
    )

    output_model_path = os.path.join(".", MODEL_BASE_FOLDER, model_setup_path + ".pt")
    trainer = utils.trainer.TrainModel(
        lr=args.learning_late,
        gpu=args.gpu_device,
        model_name=args.model,
        model_size=args.model_size,
        num_classes=len(CIFAR10_CLASSES),
    )

    output_csv_path = os.path.join(".", CSV_BASE_FOLDER, model_setup_path)

    train_loss_csv = utils.record.CSVLogSave(
        root=output_csv_path, key_name="train_loss"
    )
    train_error_csv = utils.record.CSVLogSave(
        root=output_csv_path, key_name="train_error"
    )
    train_accuracy_csv = utils.record.CSVLogSave(
        root=output_csv_path, key_name="train_accuracy"
    )

    test_loss_csv = utils.record.CSVLogSave(root=output_csv_path, key_name="test_loss")
    test_error_csv = utils.record.CSVLogSave(
        root=output_csv_path, key_name="test_error"
    )
    test_accuracy_csv = utils.record.CSVLogSave(
        root=output_csv_path, key_name="test_accuracy"
    )

    # training
    for i in range(args.epoch):
        epoch = i + 1
        train_loss, train_error, train_accuracy, model = trainer.train(train_loader)
        test_loss, test_error, test_accuracy = trainer.eval(test_loader)

        # 現在時刻
        time_now = datetime.datetime.now()
        # csvにlogを出力
        train_loss_csv.save(value=train_loss, timestamp=time_now, step=epoch)
        train_error_csv.save(value=train_error, timestamp=time_now, step=epoch)
        train_accuracy_csv.save(value=train_accuracy, timestamp=time_now, step=epoch)
        test_loss_csv.save(value=test_loss, timestamp=time_now, step=epoch)
        test_error_csv.save(value=test_error, timestamp=time_now, step=epoch)
        test_accuracy_csv.save(value=test_accuracy, timestamp=time_now, step=epoch)

    torch.save(model.state_dict(), output_model_path)


if __name__ == "__main__":
    main()
