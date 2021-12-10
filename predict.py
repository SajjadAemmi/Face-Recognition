import argparse

import cv2
import numpy as np
import torch

from backbones import get_model


@torch.no_grad()
def inference(weight, name, image_path):
    if image_path is None:
        img = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (112, 112))

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, (2, 0, 1))
    img = torch.from_numpy(img).unsqueeze(0).float()
    img.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat = net(img).numpy()
    print(feat)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch ArcFace Predict')
    parser.add_argument('--network', type=str, default='mbf', help='backbone network: for example: r50 | mbf | samnv3')
    parser.add_argument('--weight', type=str, default='weights/emore_mobilefacenet.pth')
    parser.add_argument('--input', type=str, default=None, help='aligned face image path')
    args = parser.parse_args()
    inference(args.weight, args.network, args.input)
