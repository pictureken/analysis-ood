import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
from tqdm import tqdm


class TrainModel:
    def __init__(
        self, lr: float, gpu: str, model_name: str, model_size: int, num_classes: int
    ) -> None:
        random.seed(2021)
        torch.manual_seed(2021)
        if model_name == "ResNet18":
            self.model = utils.resnet.resnet18(k=model_size, num_classes=num_classes)
        else:
            sys.exit("存在しないモデルです")
        self.num_classes = num_classes
        self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.mse = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        train_error = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            outputs = F.softmax(outputs, dim=1)
            targets_onehot = torch.eye(self.num_classes)[targets]
            error = self.mse(outputs, targets_onehot)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            train_error += error.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        train_accuracy = correct / total

        return (
            train_loss / (batch_idx + 1),
            train_error / (batch_idx + 1),
            train_accuracy,
            self.model,
        )

    def eval(self, test_loader):
        self.model.eval()
        test_loss = 0
        test_error = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in tqdm(
                enumerate(test_loader), total=len(test_loader)
            ):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                outputs = F.softmax(outputs, dim=1)
                targets_onehot = torch.eye(self.num_classes)[targets]
                error = self.mse(outputs, targets_onehot)
                test_loss += loss.item()
                test_error += error.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            test_accuracy = correct / total

        return test_loss / (batch_idx + 1), test_error / (batch_idx + 1), test_accuracy
