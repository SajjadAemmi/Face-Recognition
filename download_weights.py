# large files:
# src/models/
#           model_ir_se50.pth : 175
#           phase1_wpdc_vdc.pth.tar : 13
#           shape_predictor_68_face_landmarks.dat : 100
# src/200000_G.pth : 47
# src/weights/
#           mobilenet0.25_Final.pth : 1.8
#           Resnet50_Final.pth: 109
# src/work_space/save/model_final.pth : 175.4


import os
from urllib.request import urlretrieve
import progressbar

file_url = "https://drive.google.com/uc?id=1LGmF49EgaYhhQQtGDTiaSS2acaXaY57I"
mobilenet_url = "https://drive.google.com/uc?id=1-fIw7kZtaROkrBRITFuF0VG4sXRdrADt"
resnet_url = "https://drive.google.com/uc?id=1pubf7nOBOqhWETU-dppZTaURLS95pzyx"
G_url = "https://drive.google.com/uc?id=11OP1PqC2j-3SwYD-GkOPnkBIRYl9wL8I"
phase_1_url = "https://drive.google.com/uc?id=1J2Edy9EB1YZrUfM-UlafPF6V3VbX-ahp"
shape_predictor_url = "https://drive.google.com/uc?id=1IdBnK28DFl8QITyQMqZvQAJ4QS-zodY-"
model_ir_url = "https://drive.google.com/uc?id=1Czmrbx3QwQc991ha4wnuro9ZSdQI6nIo"
def_path = os.getcwd()


class MyProgressBar:
    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(widgets=[progressbar.Percentage(), progressbar.Bar()],
                                                maxval=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()


def download_1(file_url):
    download_dir = os.path.join('src', 'weights')
    if not os.path.exists(os.path.dirname(download_dir)):
        os.mkdir(os.path.dirname(download_dir))
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)
    # file_path = os.path.join(download_dir, 'model_final.pth')
    file_path = os.path.join(def_path,download_dir)
    if not os.path.exists(os.path.join(file_path,'model_final.pth')):
        _download_file(file_path, file_url)


def download_2(mobilenet_url, resnet_url):
    os.chdir(def_path)
    download_dir = os.path.join('src', 'weights')
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    # mobilenet_path = os.path.join(download_dir, 'mobilenet0.25_Final.pth')
    # resnet_path = os.path.join(download_dir, ' Resnet50_Final.pth')
    mobilenet_path = os.path.join(def_path,download_dir)
    resnet_path = os.path.join(def_path,download_dir)
    
    if not os.path.exists(os.path.join(mobilenet_path,'mobilenet0.25_Final.pth')):
        _download_file(mobilenet_path, mobilenet_url)
    if not os.path.exists(os.path.join(resnet_path,'Resnet50_Final.pth')):
        _download_file(resnet_path, resnet_url)


def download_3(file_url):
    os.chdir(def_path)
    download_dir = 'src'
    # file_path = os.path.join(download_dir, '200000_G.pth')
    file_path = os.path.join(def_path,download_dir)
    if not os.path.exists(os.path.join(file_path,'200000_G.pth')):
        _download_file(file_path, file_url)


def download_4(model_ir_url, phase_1_url, shape_predictor_url):
    os.chdir(def_path)
    download_dir = os.path.join('src', 'models')
    if not os.path.exists(download_dir):
        os.mkdir(download_dir)

    model_path = os.path.join(def_path, download_dir)
    if not os.path.exists(os.path.join(model_path,'model_ir_se50.pth')):
        _download_file(model_path, model_ir_url)
    if not os.path.exists(os.path.join(model_path,'phase1_wpdc_vdc.pth.tar')):
        _download_file(model_path, phase_1_url)
    if not os.path.exists(os.path.join(model_path,'shape_predictor_68_face_landmarks.dat')):
        _download_file(model_path, shape_predictor_url)


def _download_file(file_path, file_url):
    os.chdir(file_path)
    os.system('gdown '+ file_url)


if __name__ == '__main__':

    download_1(file_url)
    download_2(mobilenet_url,resnet_url)
    download_3(G_url)
    download_4(model_ir_url,phase_1_url,shape_predictor_url)
