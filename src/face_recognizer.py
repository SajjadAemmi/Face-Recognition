import numpy as np
from scipy.spatial import distance
from insightface.app import FaceAnalysis
from src.utils import *
import config


class Recognizer:
    def __init__(self, model_name):
        self.threshold = config.recognition_threshold
        self.app = FaceAnalysis(name=model_name, providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.app.prepare(ctx_id=0, det_size=(640, 640))

    def get_emb(self, face_image, tta=False):
        faces = self.app.get(face_image)
        embs = []
        bboxes = []
        for face in faces:
            embs.append(face['embedding'])
            bboxes.append(face['bbox'])
        return embs, bboxes

    @timer
    def recognize(self, image, target_embs, tta=False):
        source_embs, bboxes = self.get_emb(image)
        source_embs = np.array(source_embs)
        if len(source_embs) == 0:
            return [], []

        source_embs = np.expand_dims(source_embs, axis=2)

        target_embs = target_embs.transpose(1, 0)
        target_embs = np.expand_dims(target_embs, axis=0)

        # dist = distance.euclidean(source_embs, target_embs)
        diff = source_embs - target_embs
        dist = np.mean(np.power(diff, 2), axis=1)

        min_idx = np.argmin(dist ,axis=1)
        min_val = dist[np.arange(dist.shape[0]), min_idx]
        min_idx[min_val > self.threshold] = -1  # if no match, set idx to -1
        return min_idx, bboxes
