import argparse
import cv2
import numpy as np
from scipy.spatial import distance
from insightface.app import FaceAnalysis


parser = argparse.ArgumentParser(description='Face Identification - ArcFace with SCRFD')
parser.add_argument("--input1", default="io/input/sajjad0.jpg", type=str, help="input image 1 path")
parser.add_argument("--input2", default="io/input/sajjad1.jpg", type=str, help="input image 2 path")
args = parser.parse_args()


threshold = 20

app = FaceAnalysis(providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
app.prepare(ctx_id=0, det_size=(640, 640))
print("Model loaded")

image1 = cv2.imread(args.input1)
image2 = cv2.imread(args.input2)

faces1 = app.get(image1)
faces2 = app.get(image2)

for face1 in faces1:
    for face2 in faces2:
        dist = distance.euclidean(face1['embedding'], face2['embedding'])
        print(dist)
        if dist < threshold:
            print("Same person")
