import os
import time
import argparse

import cv2
import torch

from retina_face_detector import RetinaFaceDetector
from utils.box_utils import draw_box


parser = argparse.ArgumentParser(description='Retina Face Detector')
parser.add_argument("--input", default="input/test.jpg", help="input image path", type=str)
parser.add_argument("--output", default="output", help="output dir path", type=str)
parser.add_argument("--save", default=True, help="whether to save", action="store_true")
parser.add_argument("--origin-size", default=False, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument("--fps", default=None, type=int, help='frame per second')
parser.add_argument("--gpu", default=True, action="store_true", help='Use gpu inference')
parser.add_argument("--detection-model", default='mobilenet', help='mobilenet | resnet50')
parser.add_argument("--show_score", default=True, help="whether show the confidence score", action="store_true")
parser.add_argument("--show", default=True, help="show live result", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    detector = RetinaFaceDetector(args.detection_model, device)
    
    file_name, file_ext = os.path.splitext(os.path.basename(args.input))
    output_file_path = os.path.join(args.output, file_name + file_ext)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    tic = time.time()
    image = cv2.imread(args.input)
    if not args.origin_size:
        image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

    bounding_boxes, faces, landmarks = detector.detect(image)

    for bbox in bounding_boxes:
        draw_box(image, bbox)

    toc = time.time()

    if args.show:
        cv2.imshow('face Capture', image)
        cv2.waitKey()
    if args.save:
        cv2.imwrite(output_file_path, image)

    cv2.destroyAllWindows()
    print('finish!')
