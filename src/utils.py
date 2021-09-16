import io
import os
import functools
import time
from datetime import datetime

import numpy as np
from torchvision import transforms
import torch
import matplotlib.pyplot as plt
import cv2

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


def prepare_face_bank(detector, recognizer, device, tta=True):
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
                        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        print(file)
                    except Exception as e:
                        print(e)
                        continue

                    bounding_boxes, faces, landmarks = detector.detect(image)
                    image_face = faces[0]
            
                    emb = recognizer.get_emb(image_face, tta=True)
                    embs.append(emb)
            
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


hflip = transforms.Compose([
            de_preprocess,
            transforms.ToPILImage(),
            transforms.functional.hflip,
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])


def hflip_batch(imgs_tensor):
    hfliped_imgs = torch.empty_like(imgs_tensor)
    for i, img_ten in enumerate(imgs_tensor):
        hfliped_imgs[i] = hflip(img_ten)
    return hfliped_imgs


def get_time():
    return (str(datetime.now())[:-10]).replace(' ', '-').replace(':', '-')


def gen_plot(fpr, tpr):
    """Create a pyplot plot and save to buffer."""
    plt.figure()
    plt.xlabel("FPR", fontsize=14)
    plt.ylabel("TPR", fontsize=14)
    plt.title("ROC Curve", fontsize=14)
    plot = plt.plot(fpr, tpr, linewidth=2)
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    plt.close()
    return buf


def draw_box_name(image, bbox, name, show_score=False, score=None):
    image = cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 1)
    image = cv2.putText(image, name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)
    if show_score:
        image = cv2.putText(image, str(score), (bbox[0], bbox[3]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)
    
    return image
