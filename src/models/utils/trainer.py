import random
import sys

import torch
import torch.nn as nn
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
        self.device = torch.device(gpu if torch.cuda.is_available() else "cpu")
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def train(self, train_loader):
        self.model.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in tqdm(
            enumerate(train_loader), total=len(train_loader)
        ):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print(
            "train loss={}, train_error={}".format(
                train_loss / (batch_idx + 1), 1 - correct / total
            )
        )

    def eval(self, test_loader):
        self.model.eval()
