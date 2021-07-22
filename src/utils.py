from datetime import datetime
from PIL import Image
import numpy as np
import io
from torchvision import transforms as trans
from retina_face.data.data_pipe import de_preprocess
import torch
from src.model import l2_norm
import cv2
import os
from src.mtcnn import MTCNN
import config
import matplotlib.pyplot as plt


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


def prepare_dataset(model, device, tta=True):
    mtcnn = MTCNN()
    model.eval()
    embeddings = []
    names = ['Unknown']
    for path in config.dataset_path.iterdir():
        if path.is_file():
            continue
        else:
            embs = []
            for file in path.iterdir():
                if not file.is_file():
                    continue
                else:
                    try:
                        img = Image.open(file)
                        print(file)
                    except:
                        continue
                    if img.size != (112, 112):
                        img = mtcnn.align(img)
                    with torch.no_grad():
                        if tta:
                            mirror = trans.functional.hflip(img)
                            emb = model(config.test_transform(img).to(device).unsqueeze(0))
                            emb_mirror = model(config.test_transform(mirror).to(device).unsqueeze(0))
                            embs.append(l2_norm(emb + emb_mirror))
                        else:                        
                            embs.append(model(config.test_transform(img).to(device).unsqueeze(0)))
        if len(embs) == 0:
            continue
        embedding = torch.cat(embs).mean(0,keepdim=True)
        embeddings.append(embedding)
        names.append(path.name)
    embeddings = torch.cat(embeddings)
    names = np.array(names)
    torch.save(embeddings, os.path.join(config.face_bank_path, 'face_bank.pth'))
    np.save(config.dataset_path/'names', names)
    return embeddings, names


def load_dataset():
    embeddings = torch.load(os.path.join(config.face_bank_path, 'face_bank.pth'))
    names = np.load(os.path.join(config.face_bank_path, 'names.npy'))
    return embeddings, names


hflip = trans.Compose([
            de_preprocess,
            trans.ToPILImage(),
            trans.functional.hflip,
            trans.ToTensor(),
            trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
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
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
    cv2.putText(image, name, (bbox[0], bbox[1]), cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 0), 2, cv2.LINE_AA)
    if show_score:
        cv2.putText(image, str(score), (bbox[0], bbox[3]), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)
    
    return image
