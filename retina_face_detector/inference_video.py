import os
import time
import argparse

import cv2
import torch

from retina_face_detector import RetinaFaceDetector
from utils.box_utils import draw_box


parser = argparse.ArgumentParser(description='Retina Face Detector')
parser.add_argument("--input", default="input/obama.mp4", help="input video path", type=str)
parser.add_argument("--output", default="output", help="output dir path", type=str)
parser.add_argument("--save", default=True, help="whether to save", action="store_true")
parser.add_argument("--origin-size", default=True, type=str, help='Whether to use origin image size to evaluate')
parser.add_argument("--fps", default=None, type=int, help='frame per second')
parser.add_argument("--gpu", default=True, action="store_true", help='Use gpu inference')
parser.add_argument("--detection-model", default='mobilenet', help='mobilenet | resnet50')
parser.add_argument("--show_score", default=True, help="whether show the confidence score", action="store_true")
parser.add_argument("--show", default=True, help="show live result", action="store_true")
args = parser.parse_args()


if __name__ == '__main__':

    device = torch.device('cuda') if torch.cuda.is_available() and args.gpu else torch.device('cpu')
    detector = RetinaFaceDetector(args.detection_model, device)
    
    file_name, file_ext = os.path.splitext(os.path.basename(args.input))
    output_file_path = os.path.join(args.output, file_name + file_ext)
    
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    cap = cv2.VideoCapture(int(args.input)) if args.input.isdigit() else cv2.VideoCapture(args.input)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    cap_fps = cap.get(cv2.CAP_PROP_FPS)
    print('input video fps:', cap_fps)

    frame_rate = cap_fps // args.fps if args.fps is not None and cap_fps > args.fps else cap_fps

    if not args.origin_size:
        width = width // 2
        height = height // 2

    if args.save:
        video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_file_path, video_writer_fourcc, cap_fps, (int(width), int(height)))

    frame_count = 0
    while cap.isOpened():
        tic = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        if args.fps is not None and frame_count % frame_rate != 0:
            continue

        if not args.origin_size:
            frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        print("processing frame {} ...".format(frame_count))
        bounding_boxes, faces, landmarks = detector.detect(frame)

        for bbox in bounding_boxes:
            draw_box(frame, bbox)

        toc = time.time()
        real_fps = round(1 / (toc - tic), 4)
        frame = cv2.putText(frame, f"fps: {real_fps}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

        if args.show:
            cv2.imshow('face Capture', frame)
        if args.save:
            video_writer.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if args.save:
        video_writer.release()

    cv2.destroyAllWindows()
    print('finish!')
