import cv2
import dlib
import argparse

from src.mtcnn import MTCNN
from config import get_config, config
from src.mtcnn_pytorch.src.align_trans import get_reference_facial_points
from src.Learner import face_learner
from src.utils_main import load_facebank, prepare_facebank, draw_box_name
from retina_face.retina_face import RetinaFaceModel


class FaceRecognizer:
    def __init__(self, load_to_cpu, origin_size, update, tta):

        self.conf = get_config(False)
        self.learner_threshold = config['threshold']
        self.update = update
        self.tta = tta
        self.mtcnn = self._load_mtcnn()
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

    def _load_facebank(self):
        if self.update:
            targets, names = prepare_facebank(self.conf, self.learner.model, self.mtcnn, tta=self.tta)
            print('facebank updated')
        else:
            targets, names = load_facebank(self.conf)
            print('facebank loaded')
        return targets, names

    def generate(self, input_video, output_video, save, type, name_trackers, show):
        refrence = get_reference_facial_points(default_square=True)
        targets, names = self._load_facebank()

        camera = dict()
        camera['proj_type'] = 'orthographic'

        vidcap = cv2.VideoCapture(input_video)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = int(width // 2)
        height = int(height // 2)

        if save:
            video_writer = cv2.VideoWriter(output_video,
                                           cv2.VideoWriter_fourcc(*'mp4V'),
                                           # cv2.VideoWriter_fourcc(*'H264'),
                                           fps,
                                           (width, height))

        frame_count = 0
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))
                frame_count += 1
                print("processing frame {} ...".format(frame_count))

                bboxes_retina, faces_retina = self.retina_face.detect(frame, refrence)
                print('number of detected faces: ', len(faces_retina))

                if bboxes_retina.size == 0:
                    print("face not detected")
                    if show:
                     cv2.imshow('face Capture', frame)
                    continue

                if type == "all":
                    for bbox in bboxes_retina:
                        frame = draw_box_name(bbox, "unknown", frame)
                else:
                    results_retina, score_retina = self.learner.infer(self.conf, faces_retina, targets, self.tta)
                    # print("retina results: {}".format(results_retina))
                    # print("retina score: {}".format(score_retina))

                    for idx, bbox in enumerate(bboxes_retina):
                        name = names[results_retina[idx] + 1]
                        if name != "Unknown":
                            # print(name,name_trackers)
                            # print (name in name_trackers)
                            if name in name_trackers:
                                frame = draw_box_name(bbox, names[results_retina[idx] + 1], frame)
                            else:
                                print('not in tracker!')
                        else:
                            print('Unknown!')
                # except:
                #     count_error+=1
                #     print('detect error :'+str(count_error))

                # vis = np.concatenate((frame, origin_frame), axis=0)
                if show:
                    cv2.imshow('face Capture', frame)

                if save:
                    video_writer.write(frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        vidcap.release()
        if save:
            video_writer.release()
        cv2.destroyAllWindows()
        print('generated video saved on ', output_video)

    @staticmethod
    def _load_mtcnn():
        mtcnn = MTCNN()
        print('mtcnn loaded')
        return mtcnn

    def _load_learner(self):
        learner = face_learner(self.conf, True)
        learner.threshold = self.learner_threshold
        if self.conf.device.type == 'cpu':
            learner.load_state(self.conf, 'cpu_final.pth', True, True)
        else:
            learner.load_state(self.conf, 'final.pth', True, True)
        learner.model.eval()
        print('learner loaded')
        return learner


file_name = 'demo3.mp4'

def get_args():
    parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with RetinaFace')

    parser.add_argument('-i',
                        '--input_video',
                        help="Just the name of input video",
                        default="input/" + file_name,
                        type=str)

    parser.add_argument('-o',
                        '--output_video',
                        help="output video path",
                        default="output/" + file_name,
                        type=str)

    parser.add_argument('-ty',
                        '--type',
                        help="all | name",
                        default="name",
                        type=str)

    parser.add_argument('--origin_size',
                        default=True,
                        type=str,
                        help='Whether to use origin image size to evaluate')

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
                        help="whether perform update the facebank",
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
    return args


if __name__ == '__main__':
    args = get_args()

    face_recognizer = FaceRecognizer(load_to_cpu=args.cpu, origin_size=args.origin_size,
                                     update=args.update, tta=args.tta)

    face_recognizer.generate(input_video=args.input_video, output_video=args.output_video, save=args.save, type=args.type,
                        name_trackers=args.name_trackers, show=args.show)
