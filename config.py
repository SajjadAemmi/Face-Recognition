import os
from pathlib import Path
from torchvision import transforms as trans

# Train hyper parameters

batch_size = 64
epochs = 10
lr = 0.001
save_model = True
num_workers = 2
input_size = 112
embedding_size = 512
val = False

# Inference

weights_dir_path = './weights'

mobilenet_recognition_weights_path = os.path.join(weights_dir_path, 'model_mobilefacenet.pth')
resnet50_recognition_weights_path = os.path.join(weights_dir_path, 'model_ir_se50.pth')


confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
vis_threshold = 0.6
recognition_threshold = 1.15

data_path = Path(os.path.join('retina_face', 'data'))

input_size = [112, 112]
input_size = [256, 256]

net_depth = 50
drop_ratio = 0.6
net_mode = 'ir_se'  # or 'ir'

test_transform = trans.Compose([trans.ToTensor(),
                                       trans.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data_mode = 'emore'
vgg_folder = data_path / 'faces_vgg_112x112'
ms1m_folder = data_path / 'faces_ms1m_112x112'
emore_folder = data_path / 'faces_emore'

# Training Config
milestones = [12, 15, 18]
pin_memory = True

# Inference Config
face_bank_path = Path('./face_bank')

# the larger this value, the faster deduction, comes with tradeoff in small faces
min_face_size = 20
