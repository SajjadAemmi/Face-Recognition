import os
import gdown

if __name__ == '__main__':
    weights_path = './weights'

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    if not os.path.exists(os.path.join(weights_path, 'detection')):
        os.makedirs(os.path.join(weights_path, 'detection'))

    if not os.path.exists(os.path.join(weights_path, 'recognition')):
        os.makedirs(os.path.join(weights_path, 'recognition'))

    # detection
    mobilenet0_25_Final_url = "https://drive.google.com/uc?id=1vpRybGdvJ_c_5sF1oy_1aCgZUXtM4OV7"
    resnet50_final_url = "https://drive.google.com/uc?id=1R867SvTp2ivdeWXjFAtxgYql7RvQr3e2"

    # recognition
    model_mobilefacenet_url = "https://drive.google.com/uc?id=1SNzBx7zmmXCfXsnKtyE7wV04hQ3y1BnL"
    model_ir_se50_url = "https://drive.google.com/uc?id=1VK2HQnRt-od3Ko6nAKBS56f2HLCcGqI4"

    # download
    output = os.path.join(weights_path, 'detection', 'mobilenet0.25_Final.pth')
    gdown.download(mobilenet0_25_Final_url, output, quiet=False)

    output = os.path.join(weights_path, 'detection', 'Resnet50_Final.pth')
    gdown.download(resnet50_final_url, output, quiet=False)

    output = os.path.join(weights_path, 'recognition', 'model_mobilefacenet.pth')
    gdown.download(model_mobilefacenet_url, output, quiet=False)

    output = os.path.join(weights_path, 'recognition', 'model_ir_se50.pth')
    gdown.download(model_ir_se50_url, output, quiet=False)
