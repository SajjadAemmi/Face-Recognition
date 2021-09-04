import os
import gdown

if __name__ == '__main__':
    weights_path = './face_detector/weights'

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    mobilenet0_25_Final_url = "https://drive.google.com/uc?id=1vpRybGdvJ_c_5sF1oy_1aCgZUXtM4OV7"
    resnet50_final_url = "https://drive.google.com/uc?id=1R867SvTp2ivdeWXjFAtxgYql7RvQr3e2"

    # download
    output = os.path.join(weights_path, 'detection', 'mobilenet0.25_Final.pth')
    gdown.download(mobilenet0_25_Final_url, output, quiet=False)

    output = os.path.join(weights_path, 'detection', 'Resnet50_Final.pth')
    gdown.download(resnet50_final_url, output, quiet=False)
