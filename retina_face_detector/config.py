import os
from pathlib import Path
from torchvision import transforms


abspath = os.path.dirname(os.path.abspath(__file__))

# Inference
weights_dir_path = os.path.join(abspath, './weights')

mobilenet_detection_weights_path = os.path.join(weights_dir_path, 'mobilenet0.25_Final.pth')
resnet50_detection_weights_path = os.path.join(weights_dir_path, 'Resnet50_Final.pth')

confidence_threshold = 0.02
top_k = 5000
nms_threshold = 0.4
keep_top_k = 750
vis_threshold = 0.6

data_path = Path(os.path.join('retina_face', 'data'))

input_size = [112, 112]
input_size = [256, 256]

net_depth = 50
drop_ratio = 0.6
net_mode = 'ir_se'  # or 'ir'

test_transform = transforms.Compose([transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
data_mode = 'emore'
vgg_folder = data_path / 'faces_vgg_112x112'
ms1m_folder = data_path / 'faces_ms1m_112x112'
emore_folder = data_path / 'faces_emore'


# Inference Config
min_face_size = 20
# the larger this value, the faster deduction, comes with tradeoff in small faces
