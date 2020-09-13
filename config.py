from os.path import normpath, abspath, join, dirname, realpath
from easydict import EasyDict as edict
from pathlib import Path
import torch
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

config = {'trained_model_path': join('src', 'weights', 'mobilenet0.25_Final.pth'),
          'network_type': 'mobile0.25',
          'confidence_threshold': 0.02,
          'top_k': 5000,
          'nms_threshold': 0.4,
          'keep_top_k': 750,
          'vis_threshold': 0.5,
          'threshold': 1.15}

def get_config(training = True):
    conf = edict()

    data_path = normpath(abspath(join(dirname(realpath(__file__)), 'retina_face/data')))
    work_path = normpath(abspath(join(dirname(realpath(__file__)), 'src/work_space/')))
    conf.data_path = Path(data_path)
    conf.work_path = Path(work_path)
    conf.model_path = conf.work_path/'models'
    conf.log_path = conf.work_path/'log'
    conf.save_path = conf.work_path/'save'
    conf.input_size = [112, 112]
    conf.input_size = [256, 256]
    conf.face_landmarks_path = "./src/models/shape_predictor_68_face_landmarks.dat"

    conf.embedding_size = 512
    conf.use_mobilfacenet = False
    conf.net_depth = 50
    conf.drop_ratio = 0.6
    conf.net_mode = 'ir_se' # or 'ir'
    conf.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    conf.test_transform = trans.Compose([trans.ToTensor(),
                                         trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
    conf.data_mode = 'emore'
    conf.vgg_folder = conf.data_path/'faces_vgg_112x112'
    conf.ms1m_folder = conf.data_path/'faces_ms1m_112x112'
    conf.emore_folder = conf.data_path/'faces_emore'
    conf.batch_size = 100 # irse net depth 50 
#   conf.batch_size = 200 # mobilefacenet
#--------------------Training Config ------------------------    
    if training:        
        conf.log_path = conf.work_path/'log'
        conf.save_path = conf.work_path/'save'
    #     conf.weight_decay = 5e-4
        conf.lr = 1e-3
        conf.milestones = [12,15,18]
        conf.momentum = 0.9
        conf.pin_memory = True
#         conf.num_workers = 4 # when batchsize is 200
        conf.num_workers = 3
        conf.ce_loss = CrossEntropyLoss()    
#--------------------Inference Config ------------------------
    else:
        conf.facebank_path = conf.data_path/'facebank'
        conf.threshold = 1.5
        conf.face_limit = 10 
        #when inference, at maximum detect 10 faces in one image, my laptop is slow
        conf.min_face_size = 20
        # the larger this value, the faster deduction, comes with tradeoff in small faces
    return conf
