import os
from src.model import Backbone, Arcface, MobileFaceNet, l2_norm
import torch
import numpy as np
from tqdm import tqdm
from src.utils import *
import config


class FaceRecognizer(object):
    def __init__(self, model_name, device):
        self.threshold = config.recognition_threshold
        self.device = device

        if model_name == 'mobilenet':
            self.model = MobileFaceNet(config.embedding_size).to(self.device)
            self.model.load_state_dict(torch.load(config.mobilenet_recognition_weights_path, map_location=self.device))
        elif model_name == 'resnet50':
            self.model = Backbone(config.net_depth, config.drop_ratio, config.net_mode).to(self.device)
            self.model.load_state_dict(torch.load(config.resnet50_recognition_weights_path, map_location=self.device))
         
        self.model.eval()

    def board_val(self, db_name, accuracy, best_threshold, roc_curve_tensor):
        self.writer.add_scalar('{}_accuracy'.format(db_name), accuracy, self.step)
        self.writer.add_scalar('{}_best_threshold'.format(db_name), best_threshold, self.step)
        self.writer.add_image('{}_roc_curve'.format(db_name), roc_curve_tensor, self.step)
#         self.writer.add_scalar('{}_val:true accept ratio'.format(db_name), val, self.step)
#         self.writer.add_scalar('{}_val_std'.format(db_name), val_std, self.step)
#         self.writer.add_scalar('{}_far:False Acceptance Ratio'.format(db_name), far, self.step)
    
    def get_emb(self, image_face, tta=False):
        emb = self.model(config.test_transform(image_face).to(self.device).unsqueeze(0))
        if tta:
            mirror = cv2.flip(image_face, 1)
            emb_mirror = self.model(config.test_transform(mirror).to(self.device).unsqueeze(0))
            emb = l2_norm(emb + emb_mirror)
        return emb

    @timer
    def recognize(self, faces, target_embs, tta=False):
        embs = []
        for face in faces:
            emb = self.get_emb(face, tta)
            embs.append(emb)

        source_embs = torch.cat(embs)
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)

        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum
