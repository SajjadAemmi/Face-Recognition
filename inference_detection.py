import argparse
import cv2
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image


parser = argparse.ArgumentParser(description='Face Detection - ArcFace with SCRFD')
parser.add_argument("--input", default="input/sajjad2.jpg", type=str, help="input image 1 path")
args = parser.parse_args()


app = FaceAnalysis(providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))
img = ins_get_image('t1')
faces = app.get(img)
for face in faces:
    print(face)
rimg = app.draw_on(img, faces)
cv2.imwrite("./io/output/t1_output.jpg", rimg)
