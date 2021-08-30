import os
import gdown

if __name__ == '__main__':
    weights_path = './weights2'

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    if not os.path.exists(os.path.join(weights_path, 'detection')):
        os.makedirs(os.path.join(weights_path, 'detection'))

    if not os.path.exists(os.path.join(weights_path, 'recognition')):
        os.makedirs(os.path.join(weights_path, 'recognition'))

    onet_url = "https://drive.google.com/uc?id=1OrUwgJtEwaVERuIs6UUG3ocvma2j4H59"
    pnet_url = "https://drive.google.com/uc?id=1F6o_MYThHFSDu-DMRPgP66KRo4nbnA8f"
    rnet_url = "https://drive.google.com/uc?id=1yyRsmDyu0bRHy2_PIFMGfC5ZB7850g_b"
    shape_predictor_68_face_landmarks_url = "https://drive.google.com/uc?id=1S2j4POfDolIGdJ1nt0Aco0VpKws_c6BD"

    # detection
    mobilenet0_25_Final_url = "https://drive.google.com/uc?id=1vpRybGdvJ_c_5sF1oy_1aCgZUXtM4OV7"
    resnet50_final_url = "https://drive.google.com/uc?id=1R867SvTp2ivdeWXjFAtxgYql7RvQr3e2"

    # recognition
    model_mobilefacenet_url = "https://drive.google.com/uc?id=1SNzBx7zmmXCfXsnKtyE7wV04hQ3y1BnL"
    model_ir_se50_url = "https://drive.google.com/uc?id=1VK2HQnRt-od3Ko6nAKBS56f2HLCcGqI4"

    # download
    output = os.path.join(weights_path, 'onet.npy')
    gdown.download(onet_url, output, quiet=False)

    output = os.path.join(weights_path, 'pnet.npy')
    gdown.download(pnet_url, output, quiet=False)

    output = os.path.join(weights_path, 'rnet.npy')
    gdown.download(rnet_url, output, quiet=False)

    output = os.path.join(weights_path, 'shape_predictor_68_face_landmarks.dat')
    gdown.download(shape_predictor_68_face_landmarks_url, output, quiet=False)

    output = os.path.join(weights_path, 'detection', 'mobilenet0.25_Final.pth')
    gdown.download(mobilenet0_25_Final_url, output, quiet=False)

    output = os.path.join(weights_path, 'detection', 'Resnet50_final.pth')
    gdown.download(resnet50_final_url, output, quiet=False)

    output = os.path.join(weights_path, 'recognition', 'model_mobilefacenet.pth')
    gdown.download(model_mobilefacenet_url, output, quiet=False)

    output = os.path.join(weights_path, 'recognition', 'model_ir_se50.pth')
    gdown.download(model_ir_se50_url, output, quiet=False)
