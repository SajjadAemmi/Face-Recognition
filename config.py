from os.path import join
from easydict import EasyDict as edict
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

config = edict()

config.trained_model_path = join('weights', 'mobilenet0.25_Final.pth')
config.network_type = 'mobile0.25'
config.confidence_threshold = 0.02
config.top_k = 5000
config.nms_threshold = 0.4
config.keep_top_k = 750
config.vis_threshold = 0.5
config.recognition_threshold = 1.15

config.data_path = Path(join('retina_face', 'data'))
config.model_path = './weights'
config.input_size = [112, 112]
config.input_size = [256, 256]
config.face_landmarks_path = "./weights/shape_predictor_68_face_landmarks.dat"

config.embedding_size = 512
config.use_mobilfacenet = False
config.net_depth = 50
config.drop_ratio = 0.6
config.net_mode = 'ir_se'  # or 'ir'

config.test_transform = trans.Compose([trans.ToTensor(),
                                       trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
config.data_mode = 'emore'
config.vgg_folder = config.data_path / 'faces_vgg_112x112'
config.ms1m_folder = config.data_path / 'faces_ms1m_112x112'
config.emore_folder = config.data_path / 'faces_emore'
config.batch_size = 100  # irse net depth 50
#   config.batch_size = 200 # mobilefacenet

# Training Config
#     config.weight_decay = 5e-4
config.lr = 1e-3
config.milestones = [12, 15, 18]
config.momentum = 0.9
config.pin_memory = True
#         config.num_workers = 4 # when batchsize is 200
config.num_workers = 3
config.ce_loss = CrossEntropyLoss()

# Inference Config
config.dataset_path = Path('./dataset')
config.face_limit = 10
# when inference, at maximum detect 10 faces in one image, my laptop is slow
config.min_face_size = 20
# the larger this value, the faster deduction, comes with tradeoff in small faces
