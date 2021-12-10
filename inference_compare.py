import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, img):
    if img is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(img)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img)
    return feat


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Training')
    parser.add_argument('--network', type=str, default='mbf', help='backbone network')
    parser.add_argument('--weight', type=str, default='weights/emore_mobilefacenet.pth')
    parser.add_argument('--input1', type=str, default=None)
    parser.add_argument('--input2', type=str, default=None)
    args = parser.parse_args()
    
    emb1 = inference(args.weight, args.network, args.input1)
    emb2 = inference(args.weight, args.network, args.input2)

    diff = emb1 - emb2
    dist = torch.mean(torch.pow(diff, 2), dim=1)
    print(dist)