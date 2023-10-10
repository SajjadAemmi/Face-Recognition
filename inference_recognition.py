import os
import argparse
import mimetypes
import cv2
from src.face_recognizer import Recognizer
from src.utils import *
import config


parser = argparse.ArgumentParser(description='Face Recognition - ArcFace with SCRFD')
parser.add_argument("--input", default="io/input/IMG_5127.JPG", type=str, help="input image path")
parser.add_argument("--output", default="io/output", type=str, help="output dir path")
parser.add_argument("--update", default=False, action="store_true", help="whether perform update the dataset")
parser.add_argument("--origin-size", default=True, action="store_true", help='Whether to use origin image size to evaluate')
parser.add_argument("--tta", default=False, action="store_true", help="whether test time augmentation")
parser.add_argument("--show", default=False, action="store_true", help="show result")
parser.add_argument("--save", default=True, action="store_true", help="whether to save")
args = parser.parse_args()


if __name__ == '__main__':
    mimetypes.init()
    recognizer = Recognizer(model_name=config.model_name)

    # face bank
    if args.update:
        targets, names = prepare_face_bank(recognizer, tta=args.tta)
        print('face bank updated')
    else:
        targets, names = load_face_bank()
        print('face bank loaded')

    if args.save:
        os.makedirs(args.output, exist_ok=True)
        output_file_path = os.path.join(args.output, os.path.basename(args.input))

    if args.input.isdigit():
        webcam_is_available = True
        mimestart = None
    else:
        webcam_is_available = False
        mimestart = mimetypes.guess_type(args.input)[0]
        if mimestart == None:
            print('input not found!')
            exit()
        else:
            mimestart = mimestart.split('/')[0]

    if mimestart == 'image':
        image = cv2.imread(args.input)
        if not args.origin_size:
            image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results, bboxes = recognizer.recognize(image_rgb, targets, args.tta)

        for idx, bbox in enumerate(bboxes):
            if results[idx] != -1:
                name = names[results[idx] + 1]
            else:
                name = 'Unknown'
            image = draw_box_name(image, bbox.astype("int"), name)

        if args.show:
            cv2.imshow('face Capture', image)
            cv2.waitKey()
        if args.save:
            cv2.imwrite(output_file_path, image)

    elif mimestart == 'video' or webcam_is_available:
        cap = cv2.VideoCapture(int(args.input) if webcam_is_available else args.input)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap_fps = cap.get(cv2.CAP_PROP_FPS)
        print('input video fps:', cap_fps)

        if not args.origin_size:
            width = width // 2
            height = height // 2

        if args.save:
            video_writer_fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_writer = cv2.VideoWriter(output_file_path, video_writer_fourcc, cap_fps, (width, height))

        while cap.isOpened():
            tic = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            if not args.origin_size:
                frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results, bboxes = recognizer.recognize(frame_rgb, targets, args.tta)
            for idx, bbox in enumerate(bboxes):
                if results[idx] != -1:
                    name = names[results[idx] + 1]
                else:
                    name = 'Unknown'
                frame = draw_box_name(frame, bbox.astype("int"), name)

            toc = time.time()
            real_fps = round(1 / (toc - tic), 4)
            frame = cv2.putText(frame, f"fps: {real_fps}", (10, 20), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 0), 1, cv2.LINE_AA)

            if args.show:
                cv2.imshow('face Capture', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            if args.save:
                video_writer.write(frame)

        cap.release()
        if args.save:
            video_writer.release()
        if args.show:
            cv2.destroyAllWindows()

    print('finish!')
