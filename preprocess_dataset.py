import os
import PIL.Image as Image

import cv2
from mtcnn import MTCNN

dataset_dir_path = 'datasets/CASIA-WebFace-mini'
aligned_dataset_dir_path = 'datasets/CASIA-WebFace-mini-aligned'

mtcnn = MTCNN()

for dir in os.listdir(dataset_dir_path):
    if os.path.isdir(os.path.join(dataset_dir_path, dir)):
        print(dir)
        if not os.path.exists(os.path.join(aligned_dataset_dir_path, dir)):
            os.makedirs(os.path.join(aligned_dataset_dir_path, dir))

        for file_name in os.listdir(os.path.join(dataset_dir_path, dir)):
            if file_name.endswith('jpg') or file_name.endswith('png'):
                try:
                    image_path = os.path.join(dataset_dir_path, dir, file_name)
                    img = cv2.imread(image_path)[:, :, ::-1]
                    faces = mtcnn.align_multi(Image.fromarray(img), min_face_size=64, crop_size=(128, 128))
                    
                    for face in faces:
                        new_image_path = os.path.join(aligned_dataset_dir_path, dir, file_name)
                        print(new_image_path)
                        face.save(new_image_path)
                
                except Exception as e:
                    print(e)
                    continue
