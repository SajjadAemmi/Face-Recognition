import os
import functools
import time

import numpy as np
import torch
import cv2

from scrfd.tools.align_trans import warp_and_crop_face
import config


def de_preprocess(tensor):
    return tensor * 0.5 + 0.5


def timer(func):
    """Print the runtime of the decorated function"""
    @functools.wraps(func)
    def wrapper_timer(*args, **kwargs):
        start_time = time.perf_counter()    # 1
        value = func(*args, **kwargs)
        end_time = time.perf_counter()      # 2
        run_time = end_time - start_time    # 3
        print(f"Finished {func.__name__!r} in {run_time:.4f} secs")
        return value
    return wrapper_timer


def separate_bn_paras(modules):
    if not isinstance(modules, list):
        modules = [*modules.modules()]
    paras_only_bn = []
    paras_wo_bn = []
    for layer in modules:
        if 'model' in str(layer.__class__):
            continue
        if 'container' in str(layer.__class__):
            continue
        else:
            if 'batchnorm' in str(layer.__class__):
                paras_only_bn.extend([*layer.parameters()])
            else:
                paras_wo_bn.extend([*layer.parameters()])
    return paras_only_bn, paras_wo_bn


def prepare_face_bank(detector, recognizer, tta=True):
    embeddings = []
    names = ['Unknown']
    for path in config.face_bank_path.iterdir():
        if path.is_dir():
            print(path)  
            embs = []
            for file in path.iterdir():
                if file.is_file():
                    try:
                        image = cv2.imread(str(file))
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        print(file)
                    
                        bboxes, kpss = detector.detect(image_rgb, 0.5, input_size=(640, 640))
                        for kps in kpss:
                            face_image_aligned = warp_and_crop_face(image_rgb, kps)
                            break
                
                        emb = recognizer.get_emb(face_image_aligned, tta=tta)
                        embs.append(emb)
                        
                    except Exception as e:
                        print(e)
                        continue
            
            if len(embs) == 0:
                continue
            
            embedding = torch.cat(embs).mean(0, keepdim=True)
            embeddings.append(embedding)
            names.append(path.name)
    
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, os.path.join(config.face_bank_path, 'face_bank.pth'))
    np.save(config.face_bank_path/'names', names)
    return embeddings, names


def load_face_bank():
    embeddings = torch.load(os.path.join(config.face_bank_path, 'face_bank.pth'))
    names = np.load(os.path.join(config.face_bank_path, 'names.npy'))
    return embeddings, names


def draw_box_name(image, bbox, name, kps=None):
    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    image = cv2.putText(image, name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 0), 1, cv2.LINE_AA)
    if kps is not None:
        for kp in kps:
            kp = kp.astype(int)
            cv2.circle(image, tuple(kp) , 1, (0,0,255) , 2)

    return image
