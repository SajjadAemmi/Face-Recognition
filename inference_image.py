import sys

sys.path.insert(0, './scrfd')

import os
import time
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

from scrfd.tools.scrfd import SCRFD
from scrfd.tools.align_trans import warp_and_crop_face
from src.face_recognizer import FaceRecognizer
from src.utils import *


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with RetinaFace')
parser.add_argument("--input", default="IO/input/IMG_5127.JPG", help="input image path", type=str)
parser.add_argument("--output", default="IO/output", help="output dir path", type=str)
parser.add_argument("--save", default=True, help="whether to save", action="store_true")
parser.add_argument("--update", default=False, help="whether perform update the dataset", action="store_true")
parser.add_argument("--origin-size", default=False, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument("--gpu", action="store_true", default=True, help='Use gpu inference')
parser.add_argument("--recognition-model", default='mobilenet', help='mobilenet | resnet50')
parser.add_argument("--tta", help="whether test time augmentation", default=False, action="store_true")
parser.add_argument("--show", help="show live result", default=False, action="store_true")
# scrfd
parser.add_argument('--config', type=str, default='scrfd/configs/scrfd/scrfd_10g_bnkps.py', help='Config file')
parser.add_argument('--checkpoint', type=str, default='scrfd/weights/SCRFD_10G_KPS.pth', help='Checkpoint file')
parser.add_argument('--device', default='cuda:0', help='Device used for inference')
args = parser.parse_args()


class FaceIdentifier:
    def __init__(self, gpu, origin_size, update, tta):
        self.origin_size = origin_size
        self.tta = tta

        self.device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')
        self.detector = SCRFD(model_file='./scrfd/onnx/scrfd_10g_bnkps.onnx')
        self.recognizer = FaceRecognizer(args.recognition_model, self.device)

        # face bank
        if update:
            self.targets, self.names = prepare_face_bank(self.detector, self.recognizer, tta=self.tta)
            print('face bank updated')
        else:
            self.targets, self.names = load_face_bank()
            print('face bank loaded')
        self.targets = self.targets.to(self.device)

    @timer
    def __call__(self, input, output, save, show):
        file_name, file_ext = os.path.splitext(os.path.basename(input))
        output_file_path = os.path.join(output, file_name + file_ext)
        
        if not os.path.exists(output):
            os.makedirs(output)

        image = cv2.imread(input)

        if not self.origin_size:
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        ta = time.time()
        bboxes, kpss = self.detector.detect(image_rgb, 0.5, input_size = (640, 640))
        tb = time.time()
        print('all cost:', tb-ta)

        faces = []
        for kps in kpss:
            face_image_aligned = warp_and_crop_face(image_rgb, kps)
            faces.append(face_image_aligned)

        if len(faces) != 0:
            results = self.recognizer.recognize(faces, self.targets, self.tta)
            
            for idx, bounding_box in enumerate(bboxes):
                if results[idx] != -1:
                    name = self.names[results[idx] + 1]
                else:
                    name = 'Unknown'
                bounding_box = np.array(bounding_box, dtype="int")
                image = draw_box_name(image, bounding_box, name, kps)

        if show:
            cv2.imshow('face Capture', image)
            cv2.waitKey()
        if save:
            cv2.imwrite(output_file_path, image)

        print('finish!')


if __name__ == '__main__':
    face_identifier = FaceIdentifier(gpu=args.gpu, origin_size=args.origin_size, update=args.update, tta=args.tta)
    face_identifier(input=args.input, output=args.output, save=args.save, show=args.show)
