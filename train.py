from threading import main_thread
import time
import argparse

import torch
from torch import nn, Tensor
from torch.nn.parameter import Parameter
import torch.nn.functional as F
from tqdm import tqdm

from model import mobilenet_v3_large
from utils import calc_acc
import dataset
import config


def train():
    parser = argparse.ArgumentParser(description='SA-MobileNetV3 - Train')
    parser.add_argument('--dataset', help="dataset", default='mnist', type=str)
    parser.add_argument('--gpu', help="gpu", default=False, action='store_true')
    args = parser.parse_args()

    if args.dataset == 'mnist':
        train_dataloader, val_dataloader, dataset_classes = dataset.mnist('train')
    elif args.dataset == 'cfar100':
        train_dataloader, val_dataloader, dataset_classes = dataset.cfar100('train')

    num_classes = len(dataset_classes)

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    model = mobilenet_v3_large(num_classes=num_classes)
    model = model.to(device)

    tic = time.time()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    model.train(True)
    for epoch in range(1, config.epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        for data, labels in tqdm(train_dataloader, desc="Training"):
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            preds = model(data)
            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_acc += calc_acc(preds, labels)

        epoch_loss = train_loss / len(train_dataloader)
        epoch_acc = train_acc / len(train_dataloader)

        print(config.GREEN, f"Epoch: {epoch} [Train Loss: {epoch_loss}] [Train Accuracy: {epoch_acc}]", config.RESET)

        val_acc = 0
        for data, labels in tqdm(val_dataloader, desc="Validating"):
            data, labels = data.to(device), labels.to(device)
            preds = model(data)
            val_acc += calc_acc(preds, labels)

        acc = val_acc / len(val_dataloader)
        print(config.BLUE, f"[Validation Accuracy: {acc}]", config.RESET)


    tac = time.time()
    print("Time Taken : ", tac - tic)

    torch.save(model.state_dict(), "mnist.pt")

if __name__ == "__main__":
    train()