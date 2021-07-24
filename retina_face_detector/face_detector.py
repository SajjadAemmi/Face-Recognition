import cv2
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from .data.config import cfg_mnet, cfg_re50
from .models.retinaface import RetinaFace
from .layers.functions.prior_box import PriorBox
from .utils.box_utils import decode, decode_landm
from .utils.nms.py_cpu_nms import py_cpu_nms
from dataset_utils.mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face
import config
from src.utils import *


class FaceDetector:
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

        self.model = model
        print('RetinaNet model loaded.')

    @timer
    def detect(self, frame):
        reference = get_reference_facial_points(default_square=True)

        img = np.float32(frame)
        resize = 1
        im_height, im_width, _ = img.shape

        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.to(self.device)
        scale = scale.to(self.device)
        loc, conf, landms = self.model(img)  # forward pass

        priorbox = PriorBox(self.config, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = decode(loc.data.squeeze(0), prior_data, self.config['variance'])
        boxes = boxes * scale / resize
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
        landms = decode_landm(landms.data.squeeze(0), prior_data, self.config['variance'])
        scale = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                               img.shape[3], img.shape[2]])
        scale = scale.to(self.device)
        landms = landms * scale / resize
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
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args_retina.keep_top_k, :]
        # landms = landms[:args_retina.keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)

        faces = []
        trusted_idx = []
        landmarks = []

        for i in range(len(landms)):
            b = dets[i, :]
            if b[4] > config.vis_threshold:
                landmark = [[landms[i][2 * j], landms[i][2 * j + 1]] for j in range(5)]
                landmarks.append(landmark)
                warped_face = warp_and_crop_face(np.array(frame), landmark, reference, crop_size=(112, 112))
                
                faces.append(Image.fromarray(warped_face))
                trusted_idx.append(i)

        trusted_dets = dets[trusted_idx, :]

        return trusted_dets, faces, landmarks

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
