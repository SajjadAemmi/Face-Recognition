from os.path import join
from easydict import EasyDict as edict
from pathlib import Path
from torch.nn import CrossEntropyLoss
from torchvision import transforms as trans

# hyper parameters

batch_size = 64
epochs = 10
lr = 0.001
save_model = True
num_workers = 2
input_size = 112
embedding_size = 512
val = False

mobilenet_recognition_weights_path = join('weights', 'model_mobilefacenet.pth')
resnet50_recognition_weights_path = join('weights', 'model_ir_se50.pth')

trained_model_path = join('weights', 'mobilenet0.25_Final.pth')

confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
vis_threshold = 0.5
recognition_threshold = 1.15

data_path = Path(join('retina_face', 'data'))

model_path = './weights'

input_size = [112, 112]
input_size = [256, 256]
face_landmarks_path = "./weights/shape_predictor_68_face_landmarks.dat"

embedding_size = 512

net_depth = 50
drop_ratio = 0.6
net_mode = 'ir_se'  # or 'ir'

test_transform = trans.Compose([trans.ToTensor(),
                                       trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data_mode = 'emore'
vgg_folder = data_path / 'faces_vgg_112x112'
ms1m_folder = data_path / 'faces_ms1m_112x112'
emore_folder = data_path / 'faces_emore'
batch_size = 100  # irse net depth 50
#   batch_size = 200 # mobilefacenet

# Training Config
#     weight_decay = 5e-4
lr = 1e-3
milestones = [12, 15, 18]
momentum = 0.9
pin_memory = True
#         num_workers = 4 # when batchsize is 200
num_workers = 3
ce_loss = CrossEntropyLoss()

# Inference Config
face_bank_path = Path('./face_bank')
face_limit = 10
# when inference, at maximum detect 10 faces in one image, my laptop is slow
min_face_size = 20
# the larger this value, the faster deduction, comes with tradeoff in small faces
