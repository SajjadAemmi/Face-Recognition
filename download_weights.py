import os
import gdown

if __name__ == '__main__':
    weights_path = './weights'

    if not os.path.exists(weights_path):
        os.makedirs(weights_path)

    model_mobilefacenet_url = "https://drive.google.com/uc?id=1SNzBx7zmmXCfXsnKtyE7wV04hQ3y1BnL"
    model_ir_se50_url = "https://drive.google.com/uc?id=1VK2HQnRt-od3Ko6nAKBS56f2HLCcGqI4"

    # download
    output = os.path.join(weights_path, 'recognition', 'model_mobilefacenet.pth')
    gdown.download(model_mobilefacenet_url, output, quiet=False)

    output = os.path.join(weights_path, 'recognition', 'model_ir_se50.pth')
    gdown.download(model_ir_se50_url, output, quiet=False)
