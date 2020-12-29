import cv2
import argparse
import os
import torch

from config import config
from src.face_learner import FaceLearner
from src.utils import load_dataset, prepare_dataset, draw_box_name
from retina_face.retina_face import RetinaFaceModel


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with RetinaFace')

parser.add_argument('-i', '--input', help="input image or video path", default="input/George_Clooney.mp4", type=str)
parser.add_argument('-o', '--output', help="output image or video path", default="output/George_Clooney.mp4", type=str)
parser.add_argument('--origin_size', default=True, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument('--fps', default=None, type=int, help='frame per second')
parser.add_argument('--gpu', action="store_true", default=False, help='Use gpu inference')
parser.add_argument('--model', default='mobilenet', help='mobilenet | resnet50')
parser.add_argument("-s", "--save", help="whether to save", default=True, action="store_true")
parser.add_argument("-u", "--update", help="whether perform update the dataset", default=False, action="store_true")
parser.add_argument("-tta", "--tta", help="whether test time augmentation", default=True, action="store_true")
parser.add_argument("--show_score", help="whether show the confidence score", default=True, action="store_true")
parser.add_argument("-sh", "--show", help="show results online", default=True, action="store_true")

args = parser.parse_args()


class FaceRecognizer:
    def __init__(self, gpu, origin_size, update, tta):

        self.update = update
        self.origin_size = origin_size

        if gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.tta = tta
        self.retina_face = RetinaFaceModel(gpu, origin_size)
        self.learner = self._load_learner()
        self.targets, self.names = self._load_dataset()

    def process_image(self, image, show_score):
        bounding_boxes, faces = self.retina_face.detect(image)
        print('number of detected faces: ', len(faces))

        if len(faces) != 0:
            if type == "all":
                for bounding_box in bounding_boxes:
                    image = draw_box_name(bounding_box, "unknown", image)
            else:
                results, results_score = self.learner(faces, self.targets, self.tta)
                # print("retina results: {}".format(results))
                # print("retina score: {}".format(results_score))

                for idx, bounding_box in enumerate(bounding_boxes):
                    if results[idx] != -1:
                        name = self.names[results[idx] + 1]
                        score = round(results_score[idx].item(), 2)
                        image = draw_box_name(bounding_box, name, score, show_score, image)
                    else:
                        print('Unknown!')
        return image

    def __call__(self, input, output, save, show, show_score, fps):
        file_name, file_ext = os.path.splitext(input)

        if file_ext.lower() == '.jpg':
            image = cv2.imread(input)
            if not self.origin_size:
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

            image = self.process_image(image, show_score)
            if show:
                cv2.imshow('face Capture', image)
                cv2.waitKey()
            if save:
                cv2.imwrite(output, image)

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
                video_writer = cv2.VideoWriter(output, video_writer_fourcc, cap_fps, (int(width), int(height)))

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
            targets, names = prepare_dataset(self.learner.model, tta=self.tta)
            print('dataset updated')
        else:
            targets, names = load_dataset()
            print('dataset loaded')
        targets = targets.to(self.device)
        return targets, names

    def _load_learner(self):
        learner = FaceLearner(self.device, inference=True)
        if self.device.type == 'cpu':
            learner.load_state(config, 'cpu_final.pth', False, True)
            print('learner cpu loaded')
        else:
            learner.load_state(config, 'final.pth', False, True)
            print('learner gpu loaded')
        learner.model.eval()
        return learner


if __name__ == '__main__':
    face_recognizer = FaceRecognizer(gpu=args.gpu, origin_size=args.origin_size, update=args.update, tta=args.tta)

    face_recognizer(input=args.input, output=args.output, save=args.save, show_score=args.show_score, show=args.show,
                    fps=args.fps)
