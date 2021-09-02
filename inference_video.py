import os
import time
import argparse

import cv2
import torch
import numpy as np
from tqdm import tqdm

import config
from src.face_learner import FaceRecognizer
from src.utils import *
from retina_face_detector.face_detector import FaceDetector


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with RetinaFace')
parser.add_argument("--input", default="rtsp://192.168.1.2:8560/video_stream", help="input image or video path", type=str)
parser.add_argument("--output", default="output", help="output dir path", type=str)
parser.add_argument("--save", default=False, help="whether to save", action="store_true")
parser.add_argument("--update", default=False, help="whether perform update the dataset", action="store_true")
parser.add_argument("--origin-size", default=False, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument("--fps", default=None, type=int, help='frame per second')
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
            self.targets, self.names = prepare_face_bank(self.detector, self.recognizer, self.device, tta=self.tta)
            print('face bank updated')
        else:
            self.targets, self.names = load_face_bank()
            print('face bank loaded')
        self.targets = self.targets.to(self.device)
       
    @timer
    def __call__(self, input, output, save, show, show_score, fps):
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
       
            bounding_boxes, faces, landmarks = self.detector.detect(frame_rgb)

            if len(faces) != 0:
                results, results_score = self.recognizer.recognize(faces, self.targets, self.tta)
                for idx, bounding_box in enumerate(bounding_boxes):
                    # cv2.imshow('s', faces[idx])
                    # cv2.waitKey()

                    name = self.names[results[idx] + 1]
                    score = round(results_score[idx].item(), 2)
                    bounding_box = np.array(bounding_box, dtype="int")
                    frame = draw_box_name(frame, bounding_box, name, show_score, score)

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
    face_identifier(input=args.input, output=args.output, save=args.save, show_score=args.show_score, show=args.show, fps=args.fps)
