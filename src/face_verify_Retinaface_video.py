import cv2
from PIL import Image
import argparse
from pathlib import Path
from multiprocessing import Process, Pipe, Value, Array
import torch
from config import get_config
from mtcnn import MTCNN
from Learner import face_learner
from utils import load_facebank, draw_box_name, prepare_facebank

from mtcnn_pytorch.src.align_trans import get_reference_facial_points, warp_and_crop_face


import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import numpy as np
from data_retina import cfg_mnet, cfg_re50
from layers_retina.functions.prior_box import PriorBox
from utils_retina.nms.py_cpu_nms import py_cpu_nms
import cv2
from models_retina.retinaface import RetinaFace
from utils_retina.box_utils import decode, decode_landm
from utils_retina.timer import Timer




parser = argparse.ArgumentParser(description='Retinaface')

# parser.add_argument('-m', '--trained_model', default='./weights/mobilenet0.25_Final.pth',
#                     type=str, help='Trained state_dict file path to open')
# parser.add_argument('--network', default='mobile0.25', help='Backbone network mobile0.25 or resnet50')

parser.add_argument('-m', '--trained_model', default='./weights/Resnet50_Final.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--network', default='resnet50', help='Backbone network mobile0.25 or resnet50')

parser.add_argument('--origin_size', default=True, type=str, help='Whether use origin image size to evaluate')
parser.add_argument('--save_folder', default='./widerface_evaluate/widerface_txt/', type=str, help='Dir to save txt results')
parser.add_argument('--cpu', action="store_true", default=False, help='Use cpu inference')
parser.add_argument('--dataset_folder', default='./data/widerface/val/images/', type=str, help='dataset path')
parser.add_argument('--confidence_threshold', default=0.02, type=float, help='confidence_threshold')
parser.add_argument('--top_k', default=5000, type=int, help='top_k')
parser.add_argument('--nms_threshold', default=0.4, type=float, help='nms_threshold')
parser.add_argument('--keep_top_k', default=750, type=int, help='keep_top_k')
parser.add_argument('-s', '--save_image', action="store_true", default=True, help='show detection results')
# parser.add_argument('--vis_thres', default=0.5, type=float, help='visualization_threshold')
parser.add_argument('--vis_thres', default=0.7, type=float, help='visualization_threshold')
args_retina = parser.parse_args()


def check_keys(model, pretrained_state_dict):
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


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model

_t = {'forward_pass': Timer(), 'misc': Timer(), 'process':Timer()}



def detect_retinaface(farme,net,refrence):
    img = np.float32(frame)
    # print("Frame size is {}".format(frame.shape))
    # testing scale
    target_size = 1600
    max_size = 2150
    im_shape = img.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    resize = float(target_size) / float(im_size_min)
    # prevent bigger axis from being more than max_size:
    if np.round(resize * im_size_max) > max_size:
        resize = float(max_size) / float(im_size_max)
    if args_retina.origin_size:
        resize = 1

    if resize != 1:
        img = cv2.resize(img, None, None, fx=resize, fy=resize, interpolation=cv2.INTER_LINEAR)

    im_height, im_width, _ = img.shape
    # print("Image shape is {}".format(img.shape))

    scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
    img -= (104, 117, 123)
    img = img.transpose(2, 0, 1)
    img = torch.from_numpy(img).unsqueeze(0)
    img = img.to(device)
    scale = scale.to(device)

    _t['forward_pass'].tic()
    loc, conf, landms = net(img)  # forward pass
    _t['forward_pass'].toc()
    _t['misc'].tic()
    priorbox = PriorBox(cfg, image_size=(im_height, im_width))
    priors = priorbox.forward()
    priors = priors.to(device)
    prior_data = priors.data
    boxes = decode(loc.data.squeeze(0), prior_data, cfg['variance'])
    boxes = boxes * scale / resize
    boxes = boxes.cpu().numpy()
    scores = conf.squeeze(0).data.cpu().numpy()[:, 1]
    landms = decode_landm(landms.data.squeeze(0), prior_data, cfg['variance'])
    scale1 = torch.Tensor([img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2], img.shape[3], img.shape[2],
                           img.shape[3], img.shape[2]])
    scale1 = scale1.to(device)
    landms = landms * scale1 / resize
    landms = landms.cpu().numpy()

    # ignore low scores
    inds = np.where(scores > args_retina.confidence_threshold)[0]
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
    keep = py_cpu_nms(dets, args_retina.nms_threshold)
    # keep = nms(dets, args_retina.nms_threshold,force_cpu=args_retina.cpu)
    dets = dets[keep, :]
    landms = landms[keep]

    # keep top-K faster NMS
    # dets = dets[:args_retina.keep_top_k, :]
    # landms = landms[:args_retina.keep_top_k, :]

    # dets = np.concatenate((dets, landms), axis=1)
    _t['misc'].toc()


    print('im_detect: {:d}/{:d} forward_pass_time: {:.4f}s misc: {:.4f}s'.format(count + 1, count,
                                                                                 _t['forward_pass'].average_time,
                                                                                 _t['misc'].average_time))

    faces = []
    trusted_dets = []
    trusted_idx = []

    # for landmark in landms:
    #     facial5points = [[landmark[2*j], landmark[2*j + 1]] for j in range(5)]
    #     print(facial5points)
    #     warped_face = warp_and_crop_face(np.array(frame), facial5points, refrence, crop_size=(112, 112))
    #     cv2.imshow("Warped Face",warped_face)
    #
    #     faces.append(Image.fromarray(warped_face))

    print(len(landms))

    for idx in range(len(landms)):
        b = dets[idx,:]
        print(b[4])

        if b[4] > args_retina.vis_thres:
            landmark = landms[idx]
            facial5points = [[landmark[2*j], landmark[2*j + 1]] for j in range(5)]
            print(facial5points)
            warped_face = warp_and_crop_face(np.array(frame), facial5points, refrence, crop_size=(112, 112))
            # cv2.imshow("Warped Face",warped_face)

            faces.append(Image.fromarray(warped_face))
            trusted_idx.append(idx)

    trusted_dets = dets[trusted_idx,:]
    # print("dets is {}".format(dets))
    # print("trusted_dets is {}".format(trusted_dets))

    return trusted_dets,faces


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", default = True, action="store_true")
    # parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default= 1.2
                        , type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", default=True, action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", default = False, action="store_true")
    args = parser.parse_args()


    conf = get_config(False)
    conf.facebank_path = conf.data_path / 'facebank_2'   ######################################


    mtcnn = MTCNN()
    print('mtcnn loaded')

    refrence = get_reference_facial_points(default_square=True)

    learner = face_learner(conf, True)
    learner.threshold = args.threshold
    if conf.device.type == 'cpu':
        learner.load_state(conf, 'cpu_final.pth', True, True)
    else:
        learner.load_state(conf, 'final.pth', True, True)
    learner.model.eval()
    print('learner loaded')

    if args.update:
        targets, names = prepare_facebank(conf, learner.model, mtcnn, tta=args.tta)
        print('facebank updated')
    else:
        targets, names = load_facebank(conf)
        print('facebank loaded')

    # inital camera
    # vidcap = cv2.VideoCapture('./videos/Leonardo DiCaprio.mp4')
    # vidcap = cv2.VideoCapture('./videos/Matt Damon.mp4')
    # vidcap = cv2.VideoCapture('./videos/Matt Damon & Julianne Moore.mp4')
    # vidcap = cv2.VideoCapture('./videos/Chris Pratt Jennifer Lawrence.mp4')
    # vidcap = cv2.VideoCapture('./videos/Chris Pratt Jennifer Lawrence 2.mp4')
    # vidcap = cv2.VideoCapture('./videos/Matthew McConaughey.mp4')
    # vidcap = cv2.VideoCapture('./videos/Anelia and elle (2).mp4')  # Suitable Results
    # vidcap = cv2.VideoCapture('./videos/Julianne Moore and Michelle Williams.mp4')
    # vidcap = cv2.VideoCapture('./videos/Obama and bill.mp4')

    vidcap = cv2.VideoCapture('./videos_per/Sareh-480p__28797.mp4')


    success, image = vidcap.read()
    count = 0
    success = True

    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    width  = int(width/2.0)
    height = int(height/2.0)

    print(width,height)

    count_error = 0


    ###########################################################
    torch.set_grad_enabled(False)

    cfg = None
    if args_retina.network == "mobile0.25":
        cfg = cfg_mnet
    elif args_retina.network == "resnet50":
        cfg = cfg_re50
    # net and model
    net = RetinaFace(cfg=cfg, phase = 'test')
    net = load_model(net, args_retina.trained_model, args_retina.cpu)
    net.eval()
    print('Finished loading model!')
    print(net)
    cudnn.benchmark = True
    device = torch.device("cpu" if args_retina.cpu else "cuda")
    net = net.to(device)

    ###########################################################
    video_writer = cv2.VideoWriter('./Out_Videos.mp4', cv2.VideoWriter_fourcc(*'H264'), 25,
                                   (int(width), int(height)))

    # if args.save:
        # video_writer = cv2.VideoWriter(conf.data_path / 'recording.mp4', cv2.VideoWriter_fourcc(*'H264'), 10, (int(width), int(height)))

        # frame rate 6 due to my laptop is quite slow...
    while vidcap.isOpened():
        isSuccess, frame = vidcap.read()
        frame = cv2.resize(frame, (int(width), int(height)))
        if isSuccess:
            # try:
            #                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb


            # image = Image.fromarray(frame)
            # bboxes, faces = mtcnn.align_multi


            _t['process'].tic()


            bboxes, faces = detect_retinaface(frame,net,refrence)
            if bboxes.size ==0:
                print("Face in not detected")
                # cv2.imshow('face Capture', frame)
                continue
            # print(landmarks_retina)


            # bboxes = bboxes[:, :-1]  # shape:[10,4],only keep 10 highest possibiity faces
            # bboxes = bboxes.astype(int)
            # bboxes = bboxes + [-8, -8, 8, 8]  # personal choice

            for bbox in bboxes:
                bbox = bbox[:-1]  # shape:[10,4],only keep 10 highest possibiity faces
                bbox = bbox.astype(int)

                wid_face = bbox[2] - bbox[0]
                heigh_face = bbox[3] - bbox[1]

                cv2.rectangle(frame, (bbox[0] + int(wid_face / 2), bbox[1] + int(heigh_face / 2)),
                              (bbox[0] + int(wid_face / 2) + 5, bbox[1] + int(heigh_face / 2) + 5), (255, 255, 0),
                              4)


            results, score = learner.infer(conf, faces, targets, args.tta)
            # print("Results is {}".format(results))
            # print("Score is {}".format(score))

            for idx, bbox in enumerate(bboxes):
                if names[results[idx] + 1] != "Unknown":
                 if args.score:
                    frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                 else:
                    frame = draw_box_name(bbox, names[results[idx] + 1], frame)

            # except:
            #     count_error+=1
            #     print('detect error :'+str(count_error))

        cv2.imshow('face Capture', frame)

        if True:
            video_writer.write(frame)

        _t['process'].toc()
        print('im_saved: {:d} process: {:.4f}s'.format(count + 1, _t['process'].average_time))


        if cv2.waitKey(1) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    video_writer.release()
    cv2.destroyAllWindows()