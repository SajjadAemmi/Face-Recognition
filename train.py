import time
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
from colorama import Fore

from models import mobilenet_v3_large, Classifier
from losses import ArcFace
from utils.utils import calc_acc
from dataset_utils import dataset
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
    else:
        train_dataloader, val_dataloader, dataset_classes = None, None, None

    num_classes = len(dataset_classes)

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    model = mobilenet_v3_large(embedding_size=config.embedding_size, num_classes=num_classes).to(device)
    head = Classifier(embedding_size=config.embedding_size, num_classes=num_classes).to(device)
    arcface = ArcFace().to(device)

    tic = time.time()

    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam([{'params': model.parameters()},
                                  {'params': head.parameters()}
                                  ], lr=config.lr)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    for epoch in range(1, config.epochs + 1):
        model.train(True)
        head.train(True)
        train_loss = 0.0
        train_acc = 0.0
        for images, labels in tqdm(train_dataloader, desc="Training"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            embeddings = F.normalize(model(images))
            logits = head(embeddings)
            preds = arcface(logits, labels)

            loss = loss_fn(preds, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss
            train_acc += calc_acc(preds, labels)

        total_loss = train_loss / len(train_dataloader)
        total_acc = train_acc / len(train_dataloader)
        print(Fore.GREEN,
              f"Epoch: {epoch}",
              f"[Train Loss: {total_loss}]",
              f"[Train Accuracy: {total_acc}]",
              f"[lr: {optimizer.param_groups[0]['lr']}]",
              Fore.RESET)

        if config.val:
            model.eval()
            head.eval()
            with torch.no_grad():
                val_acc = 0
                for images, labels in tqdm(val_dataloader, desc="Validating"):
                    images, labels = images.to(device), labels.to(device)

                    embeddings = F.normalize(model(images))
                    preds = head(embeddings)

                val_acc += calc_acc(preds, labels)

                total_acc = val_acc / len(val_dataloader)
                print(Fore.BLUE,
                      f"[Validation Accuracy: {total_acc}]",
                      Fore.RESET)

        scheduler.step()

    tac = time.time()
    print("Time Taken : ", tac - tic)
    torch.save(model.state_dict(), "backbone.pt")
    torch.save(model.state_dict(), "head.pt")


if __name__ == "__main__":
    train()
