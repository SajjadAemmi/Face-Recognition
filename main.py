import os
import argparse
from pathlib import Path

import cv2
import torch
from tqdm import tqdm

import config
from src.face_learner import FaceLearner
from src.utils import load_dataset, prepare_dataset, draw_box_name
from retina_face.retina_face import RetinaFaceModel


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with RetinaFace')

parser.add_argument('-i', '--input', help="input image or video path", default="input/obama.mp4", type=str)
parser.add_argument('-o', '--output', help="output dir path", default="output", type=str)
parser.add_argument("-s", "--save", help="whether to save", default=True, action="store_true")
parser.add_argument("-u", "--update", help="whether perform update the dataset", default=False, action="store_true")
parser.add_argument('--origin_size', default=True, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument('--fps', default=None, type=int, help='frame per second')
parser.add_argument('--gpu', action="store_true", default=True, help='Use gpu inference')
parser.add_argument('--model', default='mobilenet', help='mobilenet | resnet50')
parser.add_argument("--tta", help="whether test time augmentation", default=True, action="store_true")
parser.add_argument("--show_score", help="whether show the confidence score", default=True, action="store_true")
parser.add_argument("--show", help="show live result", default=False, action="store_true")

args = parser.parse_args()


class FaceRecognizer:
    def __init__(self, gpu, origin_size, update, tta):

        self.update = update
        self.origin_size = origin_size

        self.device = torch.device('cuda') if torch.cuda.is_available() and gpu else torch.device('cpu')

        self.tta = tta
        self.detector = RetinaFaceModel(self.device, origin_size)

        self.recognizer = FaceLearner(args.model, self.device)
        self.recognizer.model.eval()

        self.targets, self.names = self._load_dataset()

    def process_image(self, image, show_score):
        bounding_boxes, faces, landmarks = self.detector.detect(image)
        print('number of detected faces: ', len(faces))

        if len(faces) != 0:
            results, results_score = self.recognizer(faces, self.targets, self.tta)
            # print("retina results: {}".format(results))
            # print("retina score: {}".format(results_score))

            for idx, bounding_box in enumerate(bounding_boxes):
                if results[idx] != -1:
                    name = self.names[results[idx] + 1]
                    score = round(results_score[idx].item(), 2)
                    image = draw_box_name(image, bounding_box, name, show_score, score)
                else:
                    print('Unknown!')
        return image

    def __call__(self, input, output, save, show, show_score, fps):
        file_name = Path(input).stem
        _, file_ext = os.path.splitext(input)
        output_file_path = os.path.join(output, file_name + file_ext)

        if not os.path.exists(output):
            os.makedirs(output)

        if file_ext.lower() == '.jpg':
            image = cv2.imread(input)
            if not self.origin_size:
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

            image = self.process_image(image, show_score)
            if show:
                cv2.imshow('face Capture', image)
                cv2.waitKey()
            if save:
                cv2.imwrite(output_file_path, image)

        elif file_ext.lower() == '.mp4' or input.isdigit():
            cap = cv2.VideoCapture(int(input)) if input.isdigit() else cv2.VideoCapture(input)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap_fps = cap.get(cv2.CAP_PROP_FPS)

            if not self.origin_size:
                width = width // 2
                height = height // 2

            if save:
                video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(output_file_path, video_writer_fourcc, cap_fps, (int(width), int(height)))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    if fps and frame_count % (cap_fps // fps) != 0:
                        continue

                    print("processing frame {} ...".format(frame_count))
                    frame = self.process_image(frame, show_score)

                    if show:
                        cv2.imshow('face Capture', frame)
                    if save:
                        video_writer.write(frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    break
            cap.release()
            if save:
                video_writer.release()
            cv2.destroyAllWindows()
        print('finish!')

    def _load_dataset(self):
        if self.update:
            targets, names = prepare_dataset(self.recognizer.model, self.device, tta=self.tta)
            print('dataset updated')
        else:
            targets, names = load_dataset()
            print('dataset loaded')
        targets = targets.to(self.device)
        return targets, names


if __name__ == '__main__':
    face_recognizer = FaceRecognizer(gpu=args.gpu, origin_size=args.origin_size, update=args.update, tta=args.tta)
    face_recognizer(input=args.input, output=args.output, save=args.save, show_score=args.show_score, show=args.show, fps=args.fps)
