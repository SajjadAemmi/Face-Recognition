import os
import argparse
import cv2
from insightface.app import FaceAnalysis


parser = argparse.ArgumentParser(description='Face Detection - ArcFace with SCRFD')
parser.add_argument("--input", default="io/input/sajjad2.jpg", type=str, help="input image 1 path")
parser.add_argument("--output", default="io/output", type=str, help="output dir path")
args = parser.parse_args()

if __name__ == '__main__':
    app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
    image = cv2.imread(args.input)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = app.get(image_rgb)
    for face in faces:
        print(face)

    result = app.draw_on(image, faces)
    output_file_path = os.path.join(args.output, os.path.basename(args.input))
    cv2.imwrite(output_file_path, result)
