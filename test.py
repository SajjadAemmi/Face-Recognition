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


def test():
    parser = argparse.ArgumentParser(description='SA-MobileNetV3 - Test')
    parser.add_argument('--dataset', help="dataset", default='mnist', type=str)
    parser.add_argument('--gpu', help="gpu", default=False, action='store_true')
    parser.add_argument('--weights', help="pre trained weights path", default='./weights/mnist.pt', type=str)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        test_dataloader, dataset_classes = dataset.mnist('test')
    elif args.dataset == 'cfar100':
        test_dataloader, dataset_classes = dataset.cfar100('test')

    num_classes = len(dataset_classes)

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    model = mobilenet_v3_large(num_classes=num_classes)
    model = model.to(device)
    model.eval()
    model.load_state_dict(torch.load(args.weights, map_location=device), strict=False)

    tic = time.time()

    model.eval()
    with torch.no_grad():
        test_acc = 0
        for data, labels in tqdm(test_dataloader, desc="Testing"):
            data, labels = data.to(device), labels.to(device)
            preds = model(data)
            test_acc += calc_acc(preds, labels)

        acc = test_acc / len(test_dataloader)
        print(config.BLUE, f"[Test Accuracy: {acc}]", config.RESET)

    tac = time.time()
    print("Time Taken : ", tac - tic)


if __name__ == "__main__":
    test()
