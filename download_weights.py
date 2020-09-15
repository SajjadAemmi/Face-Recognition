import os
import gdown


mobilenet0_25_Final_url = "https://drive.google.com/uc?id=1du0ylskUVfw6GQqmGkCnS7IxM5GD2V0T"
mobilenetV1X0_25_pretrain_url = "https://drive.google.com/uc?id=1vA_h3KxTF4kxfl7eSX8kxG92hKpQbvv8"
model_cpu_final_url = "https://drive.google.com/uc?id=1MJWsuN-BpA6qx1VefdXbeo3v-CgDf0hm"
model_final_url = "https://drive.google.com/uc?id=1Ds6B8KVU68VdJU1tICoB5VCZ7OHU7Mci"
resnet50_final_url = "https://drive.google.com/uc?id=10xlOuLk4BPw92TEJYaqmA6gidlt8TXf1"
shape_predictor_68_face_landmarks_url = "https://drive.google.com/uc?id=1z0nq0Ubf-AMrn3XMJE37IHkQaM9jk9nt"


if __name__ == '__main__':

    path = './src/weights'

    output = os.path.join(path, 'mobilenet0.25_Final.pth')
    gdown.download(mobilenet0_25_Final_url, output, quiet=False)

    output = os.path.join(path, 'mobilenetV1X0.25_pretrain.tar')
    gdown.download(mobilenetV1X0_25_pretrain_url, output, quiet=False)

    output = os.path.join(path, 'model_cpu_final.pth')
    gdown.download(model_cpu_final_url, output, quiet=False)

    output = os.path.join(path, 'model_final.pth')
    gdown.download(model_final_url, output, quiet=False)

    output = os.path.join(path, 'Resnet50_final.pth')
    gdown.download(resnet50_final_url, output, quiet=False)

    output = os.path.join(path, 'shape_predictor_68_face_landmarks.dat')
    gdown.download(shape_predictor_68_face_landmarks_url, output, quiet=False)
