import cv2
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from .data.config import cfg_mnet, cfg_re50
from .models.retinaface import RetinaFace
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms
from .utils.align_trans import get_reference_facial_points, warp_and_crop_face
from .utils.timer import timer
from . import config


class RetinaFaceDetector:
    def __init__(self, model_name, device):
        self.device = device
        self.confidence_threshold = config.confidence_threshold
        self.nms_threshold = config.nms_threshold

        torch.set_grad_enabled(False)
        if model_name == "mobilenet":
            self.trained_model_path = config.mobilenet_detection_weights_path
            self.config = cfg_mnet
        elif model_name == "resnet50":
            self.trained_model_path = config.resnet50_detection_weights_path
            self.config = cfg_re50
        
        model = RetinaFace(cfg=self.config, phase='test').to(self.device)

        pretrained_dict = torch.load(self.trained_model_path, map_location=self.device)

        if "state_dict" in pretrained_dict.keys():
            pretrained_dict = self._remove_prefix(pretrained_dict['state_dict'], 'module.')
        else:
            pretrained_dict = self._remove_prefix(pretrained_dict, 'module.')
        self._check_keys(model, pretrained_dict)
        model.load_state_dict(pretrained_dict, strict=True)
        model.eval()
        
        cudnn.benchmark = True
        self.reference = get_reference_facial_points(default_square=True)

        self.im_height, self.im_width = 640, 640
        priorbox = PriorBox(self.config, image_size=(self.im_height, self.im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        self.prior_data = priors.data

        self.model = model
        print('RetinaFace loaded')

    @timer
    def detect(self, frame):
        img = np.float32(frame)

        boxes_scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]]).to(self.device)
        landms_scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                                   img.shape[1], img.shape[0], img.shape[1], img.shape[0],
                                   img.shape[1], img.shape[0]]).to(self.device)
        
        img = cv2.resize(img, (self.im_height, self.im_width))
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).unsqueeze(0).to(self.device)
        loc, conf, landms = self.model(img)  # forward pass
   
        boxes = decode(loc.data.squeeze(0), self.prior_data, self.config['variance'])
        boxes = boxes * boxes_scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        landms = decode_landm(landms.data.squeeze(0), self.prior_data, self.config['variance'])
        landms = landms * landms_scale
        landms = landms.cpu().numpy()

        # ignore low scores
        inds = np.where(scores > self.confidence_threshold)[0]
        boxes = boxes[inds]
        landms = landms[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        # order = scores.argsort()[::-1][:self.top_k]
        boxes = boxes[order]
        landms = landms[order]
        scores = scores[order]

        # do NMS
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = py_cpu_nms(dets, self.nms_threshold)
        dets = dets[keep, :]
        landms = landms[keep]

        faces = []
        landmarks = []
        bboxes = []

        for i in range(len(landms)):
            bbox = dets[i, :]
            if bbox[4] > config.vis_threshold:
                landmark = [[landms[i][2 * j], landms[i][2 * j + 1]] for j in range(5)]
                warped_face = warp_and_crop_face(frame, landmark, self.reference, crop_size=(112, 112))
                
                landmarks.append(landmark)
                faces.append(warped_face)
                bboxes.append(bbox)

        return bboxes, faces, landmarks

    @staticmethod
    def _remove_prefix(state_dict, prefix):
        f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
        return {f(key): value for key, value in state_dict.items()}

    @staticmethod
    def _check_keys(model, pretrained_state_dict):
        ckpt_keys = set(pretrained_state_dict.keys())
        model_keys = set(model.state_dict().keys())
        used_pretrained_keys = model_keys & ckpt_keys
        assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
        return True
