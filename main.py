import cv2
import dlib
import argparse
import os

from src.mtcnn import MTCNN
from config import get_config, config
from src.mtcnn_pytorch.src.align_trans import get_reference_facial_points
from src.Learner import face_learner
from src.utils import load_dataset, prepare_dataset, draw_box_name
from retina_face.retina_face import RetinaFaceModel


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with RetinaFace')

parser.add_argument('-i',
                    '--input',
                    help="input image or video path",
                    default="input/amir_mahdi.mp4",
                    type=str)

parser.add_argument('-o',
                    '--output',
                    help="output image or video path",
                    default="output/webcam.mp4",
                    type=str)

parser.add_argument('-ty',
                    '--type',
                    help="all | name",
                    default="name",
                    type=str)

parser.add_argument('--origin_size',
                    default=False,
                    type=str,
                    help='Whether to use origin image size to evaluate')

parser.add_argument('--fps',
                    default=None,
                    type=int,
                    help='frame per second')

parser.add_argument('--cpu',
                    action="store_true",
                    default=True,
                    help='Use cpu inference')

parser.add_argument('--model',
                    default='mobilenet',
                    help='mobilenet | resnet50')

parser.add_argument('--dataset_folder',
                    default='src/data/widerface/val/images/',
                    type=str,
                    help='dataset path')

parser.add_argument("-s",
                    "--save",
                    help="whether to save",
                    default=True,
                    action="store_true")

parser.add_argument("-u",
                    "--update",
                    help="whether perform update the dataset",
                    default=False,
                    action="store_true")

parser.add_argument("-tta",
                    "--tta",
                    help="whether test time augmentation",
                    default=False,
                    action="store_true")

parser.add_argument("-c",
                    "--score",
                    help="whether show the confidence score",
                    default=True,
                    action="store_true")

parser.add_argument("-names",
                    "--name_trackers",
                    help="The person who want track",
                    type=str,
                    default="['Amir', 'Sajjad', 'Mahdi', 'Ali']")

parser.add_argument("-sh",
                    "--show",
                    help="whether perform show image results online",
                    default=True,
                    action="store_true")

args = parser.parse_args()


class FaceRecognizer:
    def __init__(self, load_to_cpu, origin_size, update, tta):

        self.conf = get_config(False)
        self.learner_threshold = config['threshold']
        self.update = update
        self.origin_size = origin_size
        self.tta = tta
        self.mtcnn = MTCNN()
        self.retina_face = RetinaFaceModel(config['network_type'],
                                           config['trained_model_path'],
                                           load_to_cpu,
                                           origin_size,
                                           config['confidence_threshold'],
                                           config['nms_threshold'],
                                           config['vis_threshold'])
        self.net = self.retina_face.load_model()
        self.learner = self._load_learner()

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.conf.face_landmarks_path)

        self.targets, self.names = self._load_dataset()
        self.reference = get_reference_facial_points(default_square=True)

    def _load_dataset(self):
        if self.update:
            targets, names = prepare_dataset(self.conf, self.learner.model, self.mtcnn, tta=self.tta)
            print('dataset updated')
        else:
            targets, names = load_dataset(self.conf)
            print('dataset loaded')
        return targets, names

    def process_image(self, image, name_trackers):
        bboxes_retina, faces_retina = self.retina_face.detect(image, self.reference)
        print('number of detected faces: ', len(faces_retina))

        if bboxes_retina.size != 0:
            if type == "all":
                for bbox in bboxes_retina:
                    frame = draw_box_name(bbox, "unknown", frame)
            else:
                results_retina, score_retina = self.learner.infer(self.conf, faces_retina, self.targets, self.tta)
                # print("retina results: {}".format(results_retina))
                # print("retina score: {}".format(score_retina))

                for idx, bbox in enumerate(bboxes_retina):
                    name = self.names[results_retina[idx] + 1]
                    if name != "Unknown":
                        if name in name_trackers:
                            image = draw_box_name(bbox, self.names[results_retina[idx] + 1], image)
                        else:
                            print('not in tracker!')
                    else:
                        print('Unknown!')
        return image

    def __call__(self, input, output, save, type, name_trackers, show, fps):
        file_name, file_ext = os.path.splitext(input)

        if file_ext.lower() == '.jpg':
            image = cv2.imread(input)
            if not self.origin_size:
                image = cv2.resize(image, (0, 0), fx=0.5, fy=0.5)

            image = self.process_image(image, name_trackers)
            if show:
                cv2.imshow('face Capture', image)
            if save:
                cv2.imwrite(output, image)

        elif file_ext.lower() == '.mp4' or input.isdigit():

            cap = cv2.VideoCapture(int(input)) if input.isdigit() else cv2.VideoCapture(input)
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            cap_fps = cap.get(cv2.CAP_PROP_FPS)

            if not self.origin_size:
                width = int(width // 2)
                height = int(height // 2)

            if save:
                video_writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*'mp4V'), cap_fps, (width, height))

            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
                    if fps and frame_count % (cap_fps // fps) != 0:
                        continue

                    print("processing frame {} ...".format(frame_count))

                    frame = self.process_image(frame, name_trackers)

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

    def _load_learner(self):
        learner = face_learner(self.conf, True)
        learner.threshold = self.learner_threshold
        if self.conf.device.type == 'cpu':
            learner.load_state(self.conf, 'cpu_final.pth', False, True)
            print('learner cpu loaded')
        else:
            learner.load_state(self.conf, 'final.pth', False, True)
            print('learner gpu loaded')
        learner.model.eval()
        return learner


if __name__ == '__main__':

    face_recognizer = FaceRecognizer(load_to_cpu=args.cpu, origin_size=args.origin_size,
                                     update=args.update, tta=args.tta)

    face_recognizer(input=args.input, output=args.output, save=args.save, type=args.type,
                    name_trackers=args.name_trackers, show=args.show, fps=args.fps)
