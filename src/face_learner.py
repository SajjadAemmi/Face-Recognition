import os
from src.model import Backbone, Arcface, MobileFaceNet, l2_norm
from src.verifacation import evaluate
import torch
import numpy as np
from tqdm import tqdm
from src.utils import *
from PIL import Image
from torchvision import transforms as trans
import math
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
        
    def evaluate(self, conf, carray, issame, nrof_folds=5, tta=False):
        self.model.eval()
        idx = 0
        embeddings = np.zeros([len(carray), config.embedding_size])
        with torch.no_grad():
            while idx + config.batch_size <= len(carray):
                batch = torch.tensor(carray[idx:idx + config.batch_size])
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(config.device)) + self.model(fliped.to(config.device))
                    embeddings[idx:idx + config.batch_size] = l2_norm(emb_batch)
                else:
                    embeddings[idx:idx + config.batch_size] = self.model(batch.to(config.device)).cpu()
                idx += config.batch_size
            if idx < len(carray):
                batch = torch.tensor(carray[idx:])            
                if tta:
                    fliped = hflip_batch(batch)
                    emb_batch = self.model(batch.to(config.device)) + self.model(fliped.to(config.device))
                    embeddings[idx:] = l2_norm(emb_batch)
                else:
                    embeddings[idx:] = self.model(batch.to(config.device)).cpu()
        tpr, fpr, accuracy, best_thresholds = evaluate(embeddings, issame, nrof_folds)
        buf = gen_plot(fpr, tpr)
        roc_curve = Image.open(buf)
        roc_curve_tensor = trans.ToTensor()(roc_curve)
        return accuracy.mean(), best_thresholds.mean(), roc_curve_tensor
    
    @timer
    def recognize(self, faces, target_embs, tta=False):
        '''
        faces : list of PIL Image
        target_embs : [n, 512] computed embeddings of faces in dataset
        names : recorded names of faces in dataset
        tta : test time augmentation (hflip, that's all)
        '''
        embs = []
        for face in faces:
            # print(img.size)
            if tta:
                mirror = trans.functional.hflip(face)
                emb = self.model(config.test_transform(face).to(self.device).unsqueeze(0))
                emb_mirror = self.model(config.test_transform(mirror).to(self.device).unsqueeze(0))
                embs.append(l2_norm(emb + emb_mirror))
            else:
                embs.append(self.model(config.test_transform(face).to(self.device).unsqueeze(0)))

        source_embs = torch.cat(embs)
        diff = source_embs.unsqueeze(-1) - target_embs.transpose(1, 0).unsqueeze(0)

        dist = torch.sum(torch.pow(diff, 2), dim=1)
        minimum, min_idx = torch.min(dist, dim=1)
        min_idx[minimum > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, minimum