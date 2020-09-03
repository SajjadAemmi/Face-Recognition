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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='for face verification')
    parser.add_argument("-s", "--save", help="whether save", default = True, action="store_true")
    # parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default=1.54, type=float)
    parser.add_argument('-th', '--threshold', help='threshold to decide identical faces', default= 1.6
                        , type=float)
    parser.add_argument("-u", "--update", help="whether perform update the facebank", default=True, action="store_true")
    parser.add_argument("-tta", "--tta", help="whether test time augmentation", action="store_true")
    parser.add_argument("-c", "--score", help="whether show the confidence score", default = False, action="store_true")
    args = parser.parse_args()

    conf = get_config(False)

    mtcnn = MTCNN()
    print('mtcnn loaded')

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
    vidcap = cv2.VideoCapture('./videos/Matt Damon.mp4')
    # vidcap = cv2.VideoCapture('./videos/Matt Damon & Julianne Moore.mp4')
    # vidcap = cv2.VideoCapture('./videos/Chris Pratt Jennifer Lawrence.mp4')
    # vidcap = cv2.VideoCapture('./videos/Chris Pratt Jennifer Lawrence 2.mp4')
    # vidcap = cv2.VideoCapture('./videos/Matthew McConaughey.mp4')
    # vidcap = cv2.VideoCapture('./videos/Anelia and elle (2).mp4')  # Suitable Results
    # vidcap = cv2.VideoCapture('./videos/Julianne Moore and Michelle Williams.mp4')
    # vidcap = cv2.VideoCapture('./videos/Obama and bill.mp4')


    success, image = vidcap.read()
    count = 0
    success = True

    width = vidcap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    width=  int(width//2)
    height = int(height//2)

    count_error = 0

    if args.save:
        # video_writer = cv2.VideoWriter(conf.data_path / 'recording.mp4', cv2.VideoWriter_fourcc(*'H264'), 10, (int(width), int(height)))
        video_writer = cv2.VideoWriter('./Out_Videos.mp4', cv2.VideoWriter_fourcc(*'H264'), float(25), (int(width), int(height)))
        # frame rate 6 due to my laptop is quite slow...
    while vidcap.isOpened():
        isSuccess, frame = vidcap.read()
        frame = cv2.resize(frame, (width, height))
        if isSuccess:
            try:
                #                 image = Image.fromarray(frame[...,::-1]) #bgr to rgb
                image = Image.fromarray(frame)
                bboxes, faces = mtcnn.align_multi(image, conf.face_limit, conf.min_face_size)
                print("Boxes is {}".format(bboxes))

                bboxes = bboxes[:, :-1]    # shape:[10,4],only keep 10 highest possibiity faces
                bboxes = bboxes.astype(int)
                bboxes = bboxes + [-8, -8, 8, 8]  # personal choice

                for bbox in bboxes:
                    wid_face = bbox[2] - bbox[0]
                    heigh_face = bbox[3] - bbox[1]
                    cv2.rectangle(frame, (bbox[0] + int(wid_face / 2), bbox[1] + int(heigh_face / 2)),
                                  (bbox[0] + int(wid_face / 2) + 5, bbox[1] + int(heigh_face / 2) + 5), (255, 255, 0),
                                  4)

                results, score = learner.infer(conf, faces, targets, args.tta)
                print("Results is {}".format(results))
                print("Score is {}".format(score))

                for idx, bbox in enumerate(bboxes):
                    if names[results[idx] + 1] != "Unknown":
                     if args.score:
                        frame = draw_box_name(bbox, names[results[idx] + 1] + '_{:.2f}'.format(score[idx]), frame)
                     else:
                        frame = draw_box_name(bbox, names[results[idx] + 1], frame)
            except:
                count_error+=1
                print('detect error :'+str(count_error))

            cv2.imshow('face Capture', frame)

        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    vidcap.release()
    if args.save:
        video_writer.release()
    cv2.destroyAllWindows()