import os
import gdown

path = './weights'

if not os.path.exists(path):
    os.makedirs(path)

if not os.path.exists(os.path.join(path, 'detecion')):
    os.makedirs(os.path.join(path, 'detecion'))

if not os.path.exists(os.path.join(path, 'recognition')):
    os.makedirs(os.path.join(path, 'recognition'))

mobilenet0_25_Final_url = "https://drive.google.com/uc?id=1du0ylskUVfw6GQqmGkCnS7IxM5GD2V0T"
resnet50_final_url = "https://drive.google.com/uc?id=10xlOuLk4BPw92TEJYaqmA6gidlt8TXf1"
shape_predictor_68_face_landmarks_url = "https://drive.google.com/uc?id=1z0nq0Ubf-AMrn3XMJE37IHkQaM9jk9nt"

if __name__ == '__main__':
    output = os.path.join(os.path.join(path, 'detecion'), 'mobilenet0.25_Final.pth')
    gdown.download(mobilenet0_25_Final_url, output, quiet=False)

    output = os.path.join(os.path.join(path, 'detecion'), 'Resnet50_final.pth')
    gdown.download(resnet50_final_url, output, quiet=False)

    output = os.path.join(path, 'shape_predictor_68_face_landmarks.dat')
    gdown.download(shape_predictor_68_face_landmarks_url, output, quiet=False)
