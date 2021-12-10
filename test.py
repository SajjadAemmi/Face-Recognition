import time
import argparse

import torch
import torch.nn.functional as F
from tqdm import tqdm
from colorama import Fore

from models import mobilenet_v3_large, Classifier
from utils.utils import calc_acc
from dataset_utils import dataset
import config


def test():
    parser = argparse.ArgumentParser(description='SA-MobileNetV3 - Test')
    parser.add_argument('--dataset', help="dataset", default='mnist', type=str)
    parser.add_argument('--gpu', help="gpu", default=False, action='store_true')
    parser.add_argument('--backbone-weights', help="pre trained weights path", default='./weights/backbone.pt', type=str)
    parser.add_argument('--head-weights', help="pre trained weights path", default='./weights/head.pt', type=str)
    args = parser.parse_args()

    if args.dataset == 'mnist':
        test_dataloader, dataset_classes = dataset.mnist('test')
    elif args.dataset == 'cfar100':
        test_dataloader, dataset_classes = dataset.cfar100('test')
    else:
        test_dataloader, dataset_classes = None, None

    num_classes = len(dataset_classes)

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    model = mobilenet_v3_large(embedding_size=config.embedding_size, num_classes=num_classes).to(device)
    model.load_state_dict(torch.load(args.backbone_weights, map_location=device), strict=False)
    model.eval()
    head = Classifier(embedding_size=config.embedding_size, num_classes=num_classes).to(device)
    head.load_state_dict(torch.load(args.head_weights, map_location=device), strict=False)
    head.eval()

    tic = time.time()

    with torch.no_grad():
        test_acc = 0
        for images, labels in tqdm(test_dataloader, desc="Testing"):
            images, labels = images.to(device), labels.to(device)
            embeddings = F.normalize(model(images))
            preds = head(embeddings)

        test_acc += calc_acc(preds, labels)

        acc = test_acc / len(test_dataloader)
        print(Fore.BLUE,
              f"[Test Accuracy: {acc}]",
              Fore.RESET)

    tac = time.time()
    print("Time Taken : ", tac - tic)


if __name__ == "__main__":
    test()
