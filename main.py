import argparse

from Face_Tracker import FaceTracker
from config_main import config

file_name = 'amir_sajjad.mp4'

def get_args():
    parser = argparse.ArgumentParser(description='Face Recognition - Arcface with RetinaFace')

    parser.add_argument('-i',
                        '--input_video',
                        help="Just the name of input video",
                        default="input/" + file_name,
                        type=str)

    parser.add_argument('-o',
                        '--output_video',
                        help="output video path",
                        default="output/" + file_name,
                        type=str)

    parser.add_argument('-ty',
                        '--type',
                        help="all | name",
                        default="name",
                        type=str)

    parser.add_argument('--origin_size',
                        default=True,
                        type=str,
                        help='Whether to use origin image size to evaluate')

    parser.add_argument('--cpu',
                        action="store_true",
                        default=True,
                        help='Use cpu inference')

    parser.add_argument('--model',
                        default='mobilenet',
                        help='mobilenet | resnet50')

    parser.add_argument('--dataset_folder',
                        default='src/data/widerface/val/images/',
                        type=str,
                        help='dataset path')

    parser.add_argument("-s",
                        "--save",
                        help="whether to save",
                        default=True,
                        action="store_true")

    parser.add_argument("-u",
                        "--update",
                        help="whether perform update the facebank",
                        default=False,
                        action="store_true")

    parser.add_argument("-tta",
                        "--tta",
                        help="whether test time augmentation",
                        action="store_true")

    parser.add_argument("-c",
                        "--score",
                        help="whether show the confidence score",
                        default=False,
                        action="store_true")

    parser.add_argument("-names",
                        "--name_trackers",
                        help="The person who want track",
                        type=str,
                        default="['Amir', 'Sajjad']")

    parser.add_argument("-sh",
                        "--show",
                        help="whether perform show image results online",
                        default=True,
                        action="store_true")

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = get_args()

    face_anony = FaceTracker(config=config, load_to_cpu=args.cpu, origin_size=args.origin_size,
                             update=args.update, tta=args.tta)

    face_anony.generate(input_video=args.input_video, output_video=args.output_video, save=args.save, type=args.type,
                        name_trackers=args.name_trackers, show=args.show)
