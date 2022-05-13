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


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with SCRFD')
parser.add_argument("--input", default="0", help="input image or video path", type=str)
parser.add_argument("--output", default="IO/output", help="output dir path", type=str)
parser.add_argument("--save", default=True, help="whether to save", action="store_true")
parser.add_argument("--update", default=False, help="whether perform update the dataset", action="store_true")
parser.add_argument("--origin-size", default=True, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument("--fps", default=None, type=int, help='frame per second')
parser.add_argument("--gpu", action="store_true", default=True, help='Use gpu inference')
parser.add_argument("--detection-model", default='resnet50', help='mobilenet | resnet50')
parser.add_argument("--recognition-model", default='resnet50', help='mobilenet | resnet50')
parser.add_argument("--tta", help="whether test time augmentation", default=False, action="store_true")
parser.add_argument("--show", help="show live result", default=False, action="store_true")
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
    def __call__(self, input, output, save, show, fps):
        file_name, file_ext = os.path.splitext(os.path.basename(input))
        output_file_path = os.path.join(output, file_name + file_ext)
        
        if not os.path.exists(output):
            os.makedirs(output)

        cap = cv2.VideoCapture(int(input)) if input.isdigit() else cv2.VideoCapture(input)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        print('input video fps:', cap_fps)

        frame_rate = cap_fps // fps if fps is not None and cap_fps > fps else cap_fps

        if not self.origin_size:
            width = width // 2
            height = height // 2

        if save:
            video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file_path, video_writer_fourcc, cap_fps, (int(width), int(height)))

        frame_count = 0
        while cap.isOpened():
            tic = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1
            if fps is not None and frame_count % frame_rate != 0:
                continue

            if not self.origin_size:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            print("processing frame {} ...".format(frame_count))
       
            ta = time.time()
            bboxes, kpss = self.detector.detect(frame_rgb, 0.5, input_size = (640, 640))
            tb = time.time()
            print('detection time:', tb-ta)

            faces = []
            for kps in kpss:
                face_image_aligned = warp_and_crop_face(frame_rgb, kps)
                faces.append(face_image_aligned)

            if len(faces) != 0:
                results = self.recognizer.recognize(faces, self.targets, self.tta)
                for idx, bbox in enumerate(bboxes):
                    if results[idx] != -1:
                        name = self.names[results[idx] + 1]
                    else:
                        name = 'Unknown'
                    bbox = np.array(bbox, dtype="int")
                    frame = draw_box_name(frame, bbox, name)

            toc = time.time()
            real_fps = round(1 / (toc - tic), 4)
            frame = cv2.putText(frame, f"fps: {real_fps}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

            if show:
                cv2.imshow('face Capture', frame)
            if save:
                video_writer.write(frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        if save:
            video_writer.release()

        cv2.destroyAllWindows()
        print('finish!')


if __name__ == '__main__':
    face_identifier = FaceIdentifier(gpu=args.gpu, origin_size=args.origin_size, update=args.update, tta=args.tta)
    face_identifier(input=args.input, output=args.output, save=args.save, show=args.show, fps=args.fps)
