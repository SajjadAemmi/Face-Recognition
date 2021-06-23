import cv2
from PIL import Image
import torch
import torch.backends.cudnn as cudnn
import numpy as np

from retina_face.data.config import cfg_mnet, cfg_re50
from retina_face.models.retinaface import RetinaFace
from retina_face.layers.functions.prior_box import PriorBox
from retina_face.utils.box_utils import decode, decode_landm
from retina_face.utils.nms.py_cpu_nms import py_cpu_nms
from mtcnn_pytorch.src.align_trans import warp_and_crop_face
from mtcnn_pytorch.src.align_trans import get_reference_facial_points
from config import config


class RetinaFaceModel:
    def __init__(self, gpu, origin_size):
        self.trained_model_path = config.trained_model_path
        self.gpu = gpu
        self.origin_size = origin_size
        self.confidence_threshold = config.confidence_threshold
        self.nms_threshold = config.nms_threshold
        self.cfg = None
        self.device = None
        self.net = None
        self.load_model()

    def load_model(self):
        torch.set_grad_enabled(False)
        if config.network_type == "mobile0.25":
            self.cfg = cfg_mnet
        elif config.network_type == "resnet50":
            self.cfg = cfg_re50
        model = RetinaFace(cfg=self.cfg, phase='test')
        print('Loading pretrained model from {}'.format(self.trained_model_path))
        if self.gpu:
            device = torch.cuda.current_device()
            pretrained_dict = torch.load(self.trained_model_path, map_location=lambda storage, loc: storage.cuda(device))
        else:
            pretrained_dict = torch.load(self.trained_model_path, map_location=torch.device('cpu'))

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

        if self.gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        model = model.to(self.device)
        self.net = model

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
        loc, conf, landms = self.net(img)  # forward pass

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
        dets = dets[keep, :]
        landms = landms[keep]

        # keep top-K faster NMS
        # dets = dets[:args_retina.keep_top_k, :]
        # landms = landms[:args_retina.keep_top_k, :]

        # dets = np.concatenate((dets, landms), axis=1)

        faces = []
        trusted_idx = []

        for idx in range(len(landms)):
            b = dets[idx, :]
            # print(b)

            if b[4] > config.vis_threshold:
                landmark = landms[idx]
                facial5points = [[landmark[2 * j], landmark[2 * j + 1]] for j in range(5)]
                # print(facial5points)
                warped_face = warp_and_crop_face(np.array(frame), facial5points, reference, crop_size=(112, 112))
                # cv2.imshow("Warped Face", warped_face)

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
