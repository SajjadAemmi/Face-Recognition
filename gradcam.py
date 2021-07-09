import os
import time
import argparse
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from colorama import Fore

from models import mobilenet_v3_large, Classifier
from src.model import MobileFaceNet, Backbone
import config


def gradcam(image_path):
    image = cv2.imread(image_path)
    image_tensor = transform(image).unsqueeze(0).to(device)

    cam = GradCAM(model=model, target_layer=target_layer, use_cuda=args.gpu)

    input_tensor = image_tensor.float()
    grayscale_cam = cam(input_tensor=input_tensor, target_category=None, aug_smooth=True)
    grayscale_cam = grayscale_cam[0]

    image = cv2.resize(image, (config.input_size, config.input_size))
    image_normal = image / 255.0

    visualization = show_cam_on_image(image_normal, grayscale_cam)

    output_path = os.path.join(args.output_dir, Path(image_path).stem + "_" + args.net + ".jpg")
    cv2.imwrite(output_path, visualization)


def run():
    model.eval()

    tic = time.time()

    if os.path.isfile(args.input):
        gradcam(args.input)
    elif os.path.isdir(args.input):
        for file_name in os.listdir(args.input):
            if os.path.isfile(os.path.join(args.input, file_name)):
                print(file_name)
                gradcam(os.path.join(args.input, file_name))

    tac = time.time()
    print("Time Taken : ", tac - tic)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='SA-MobileNetV3 - Predict')
    parser.add_argument('--input', default='input/aligned_faces_occlusion', help="input image or dir", type=str)
    parser.add_argument('--output-dir', default='output', help="output dir path", type=str)
    parser.add_argument('--dataset', default='mnist', help="dataset", type=str)
    parser.add_argument('--gpu', default=False, help="gpu", action='store_true')
    parser.add_argument('--net', default='SA-MobileNetV3', help="SA-MobileNetV3 | MobileFaceNet | ResNet50", type=str)
    parser.add_argument('--weights-dir', default='weights', help="pretrained weights dir path", type=str)
    args = parser.parse_args()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((config.input_size, config.input_size))
                                    #  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    #                       std=[0.229, 0.224, 0.225]),
                                    ])

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')

    if args.net == 'SA-MobileNetV3':
        weights_path = os.path.join(args.weights_dir, 'model_sa_mobilenetv3.pth')
        model = mobilenet_v3_large(embedding_size=config.embedding_size).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
        target_layer = model.features[-1]
    elif args.net == 'MobileFaceNet':
        weights_path = os.path.join(args.weights_dir, 'model_mobilefacenet.pth')
        model = MobileFaceNet(config.embedding_size).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
        target_layer = model.conv_5
    elif args.net == 'ResNet50':
        weights_path = os.path.join(args.weights_dir, 'model_ir_se50.pth')
        model = Backbone(config.config.net_depth, config.config.drop_ratio, config.config.net_mode).to(device)
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=True)
        target_layer = model.body[22]

    run()
