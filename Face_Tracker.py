import numpy as np
import cv2
import scipy.io as sio
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import dlib

from src.mtcnn import MTCNN
from config import get_config
from src.data_retina import cfg_mnet, cfg_re50
from src.models_retina.retinaface import RetinaFace
from src.utils_retina.timer import Timer
from src.layers_retina.functions.prior_box import PriorBox
from src.utils_retina.box_utils import decode, decode_landm
from src.utils_retina.nms.py_cpu_nms import py_cpu_nms
from src.mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
from src.Learner import face_learner
from src.utils_main import load_facebank, prepare_facebank, draw_box_name


class FaceTracker:

    def __init__(self, config, load_to_cpu, origin_size, update, tta):

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

        # static declareation for path
        self.model = "./src/models/shape_predictor_68_face_landmarks.dat"
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(self.model)

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

        pose = [0, 0, 0]
        camera = dict()
        camera['proj_type'] = 'orthographic'

        vidcap = cv2.VideoCapture(input_video)
        width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = vidcap.get(cv2.CAP_PROP_FPS)
        width = int(width // 2)
        height = int(height // 2)

        if save:
            if output_video.endswith('mp4'):
                fourcc = cv2.VideoWriter_fourcc(*'mp4V')
            else:
                fourcc = cv2.VideoWriter_fourcc(*'MPEG')
            video_writer = cv2.VideoWriter(output_video,
                                           fourcc,
                                           # cv2.VideoWriter_fourcc(*'H264'),
                                           fps,
                                           (width, height))
            print(video_writer)
            # frame rate 6 due to my laptop is quite slow...

        frame_count = 0
        while vidcap.isOpened():
            ret, frame = vidcap.read()
            if ret:
                frame = cv2.resize(frame, (width, height))
                frame_count += 1
                print("processing frame {} ...".format(frame_count))

                # image = Image.fromarray(frame)
                # try:
                #     bboxes, faces = self.mtcnn.align_multi(image,
                #                                            self.conf.face_limit,
                #                                            self.conf.min_face_size)
                # except:
                #     faces = []
                #     bboxes = []

                # print(faces)
                # print("number of detected faces: {}".format(len(faces)))

                bboxes_retina, faces_retina = self.retina_face.detect(frame, refrence)
                print('number of detected faces: ', len(faces_retina))

                if bboxes_retina.size == 0:
                    print("face not detected")
                    if show:
                     cv2.imshow('face Capture', frame)
                    continue
                # print(landmarks_retina)

                # bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
                # bboxes = bboxes.astype(int)
                # bboxes = bboxes + [-8, -8, 8, 8]  # personal choice

                if type == "all":

                 for bbox in bboxes_retina:
                     frame = draw_box_name(bbox, "unknown", frame)

                else:

                 results_retina, score_retina = self.learner.infer(self.conf, faces_retina, targets, self.tta)

                 print("retina results: {}".format(results_retina))
                 print("retina score: {}".format(score_retina))

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

    # @staticmethod
    # def _process_image(image_input):
    #     make_3D_fromImage(image_input)
    #     Image_filename = str(image_input)
    #     suffix = Image_filename[-4::]
    #
    #     # print(suffix)
    #     print(Image_filename.replace(suffix, ''))
    #
    #     model3D_mesh_1 = sio.loadmat('{}_0_new.mat'.format(Image_filename.replace(suffix, '')))
    #
    #     vertices = model3D_mesh_1["vertices"]
    #     triangles = model3D_mesh_1["triangles"]
    #     colors = model3D_mesh_1["colors"]
    #
    #     return vertices, triangles, colors


class RetinaFaceModel:

    def __init__(self,
                 network_type,
                 trained_model_path,
                 cpu,
                 origin_size,
                 confidence_threshold,
                 nms_threshold,
                 vis_threshold):
        self.network_type = network_type
        self.trained_model_path = trained_model_path
        self.load_to_cpu = cpu
        self.origin_size = origin_size
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.vis_threshold = vis_threshold
        self.cfg = None
        self.device = None
        self.net = None

    def load_model(self):
        torch.set_grad_enabled(False)
        if self.network_type == "mobile0.25":
            self.cfg = cfg_mnet
        elif self.network_type == "resnet50":
            self.cfg = cfg_re50
        model = RetinaFace(cfg=self.cfg, phase='test')
        print('Loading pretrained model from {}'.format(self.trained_model_path))
        if self.load_to_cpu:
            pretrained_dict = torch.load(self.trained_model_path, map_location=torch.device('cpu'))
        else:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(self.trained_model_path, map_location=lambda storage, loc: storage.cuda(device))

        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=False)
        model.eval()
        print('RetinaNet model loaded.')
        # print(model)
        cudnn.benchmark = True
        device = torch.device("cpu" if self.load_to_cpu else "cuda")
        model = model.to(device)
        self.device = device
        self.net = model
        return model

    def detect(self,
               frame,
               refrence,
               target_size=1600,
               max_size=2150):
        img = np.float32(frame)
        # testing scale
        im_shape = img.shape
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > max_size:
            resize = float(max_size) / float(im_size_max)
        if self.origin_size:
            resize = 1

        if resize != 1:
            img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

        im_height, im_width, _ = img.shape
        # print("Image shape is {}".format(img.shape))

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)

        _t = {'forward_pass': Timer(), 'misc': Timer()}
        _t['forward_pass'].tic()
        loc, conf, landms = self.net(img)  # forward pass
        _t['forward_pass'].toc()
        _t['misc'].tic()
        priorbox = PriorBox(self.cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.cfg['variance'])
        scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale1 = scale1.to(self.device)
        landms = landms * scale1 / resize
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:args_retina.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        # keep = nms(dets, args_retina.nms_threshold,force_cpu=args_retina.cpu)
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args_retina.keep_top_k, :]
        # landms = landms[:args_retina.keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)
        _t['misc'].toc()

        print('forward_pass_time: {:.4f}s misc: {:.4f}s'.format(_t['forward_pass'].average_time,
                                                                _t['misc'].average_time))

        faces = []
        trusted_idx = []

        # for landmark in landms:
        #     facial5points = [[landmark[2*j], landmark[2*j + 1]] for j in range(5)]
        #     print(facial5points)
        #     warped_face = warp_and_crop_face(np.array(frame), facial5points, refrence, crop_size=(112, 112))
        #     cv2.imshow("Warped Face",warped_face)
        #
        #     faces.append(Image.fromarray(warped_face))

        # print(len(landms))

        for idx in range(len(landms)):
            b = dets[idx, :]
            print(b[4])

            if b[4] > self.vis_threshold:
                landmark = landms[idx]
                facial5points = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]
                # print(facial5points)
                warped_face = warp_and_crop_face(np.array(frame), facial5points, refrence, crop_size=(112, 112))
                # cv2.imshow("Warped Face 2",warped_face)

                faces.append(Image.fromarray(warped_face))
                trusted_idx.append(idx)

        trusted_dets = dets[trusted_idx, :]
        # print("dets is {}".format(dets))
        # print("trusted_dets is {}".format(trusted_dets))

        return trusted_dets, faces

    @staticmethod
    def _remove_prefix(state_dict, prefix):
        ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
        print('remove prefix \'{}\''.format(prefix))
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    @staticmethod
    def _check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        unused_pretrained_keys = ckpt_keys - model_keys
        missing_keys = model_keys - ckpt_keys
        print('Missing keys:{}'.format(len(missing_keys)))
        print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
        print('Used keys:{}'.format(len(used_pretrained_keys)))
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True
