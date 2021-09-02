import os
import time
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

import config
from src.face_recognizer import FaceRecognizer
from src.utils import *
from retina_face_detector.face_detector import FaceDetector


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with RetinaFace')
parser.add_argument("--input", default="0", help="input image or video path", type=str)
parser.add_argument("--output", default="output", help="output dir path", type=str)
parser.add_argument("--save", default=True, help="whether to save", action="store_true")
parser.add_argument("--update", default=False, help="whether perform update the dataset", action="store_true")
parser.add_argument("--origin-size", default=False, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument("--gpu", action="store_true", default=True, help='Use gpu inference')
parser.add_argument("--detection-model", default='mobilenet', help='mobilenet | resnet50')
parser.add_argument("--recognition-model", default='mobilenet', help='mobilenet | resnet50')
parser.add_argument("--tta", help="whether test time augmentation", default=False, action="store_true")
parser.add_argument("--show_score", help="whether show the confidence score", default=True, action="store_true")
parser.add_argument("--show", help="show live result", default=True, action="store_true")
args = parser.parse_args()


class FaceIdentifier:
    def __init__(self, gpu, origin_size, update, tta):
        self.origin_size = origin_size
        self.tta = tta

        self.device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')
        self.detector = FaceDetector(args.detection_model, self.device)
        self.recognizer = FaceRecognizer(args.recognition_model, self.device)

        # face bank
        if update:
            self.targets, self.names = prepare_face_bank(self.recognizer.model, self.device, tta=self.tta)
            print('face bank updated')
        else:
            self.targets, self.names = load_face_bank()
            print('face bank loaded')
        self.targets = self.targets.to(self.device)

    @timer
    def __call__(self, input, output, save, show, show_score):
        file_name, file_ext = os.path.splitext(os.path.basename(input))
        output_file_path = os.path.join(output, file_name + file_ext)
        
        if not os.path.exists(output):
            os.makedirs(output)

        image = cv2.imread(input)

        if not self.origin_size:
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        bounding_boxes, faces, landmarks = self.detector.detect(image_rgb)

        if len(faces) != 0:
            results, results_score = self.recognizer.recognize(faces, self.targets, self.tta)
            for idx, bounding_box in enumerate(bounding_boxes):
                if results[idx] != -1:
                    name = self.names[results[idx] + 1]
                else:
                    name = 'Unknown'
                score = round(results_score[idx].item(), 2)
                bounding_box = np.array(bounding_box, dtype="int")
                image = draw_box_name(image, bounding_box, name, show_score, score)

        if show:
            cv2.imshow('face Capture', image)
            cv2.waitKey()
        if save:
            cv2.imwrite(output_file_path, image)

        print('finish!')


if __name__ == '__main__':
    face_identifier = FaceIdentifier(gpu=args.gpu, origin_size=args.origin_size, update=args.update, tta=args.tta)
    face_identifier(input=args.input, output=args.output, save=args.save, show_score=args.show_score, show=args.show)
