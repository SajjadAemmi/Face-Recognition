from os.path import normpath, abspath, join, dirname, realpath


##############################################################
# trained_model_path: Trained state_dict file path to open
# network_type: Backbone network mobile0.25 or resnet50
# confidence_threshold: confidence_threshold
# top_k: top_k
# nms_threshold: nms_threshold
# keep_top_k: keep_top_k
# vis_threshold: visualization threshold
# threshold: threshold to decide identical faces


config = {'trained_model_path': join('src', 'weights', 'mobilenet0.25_Final.pth'),
          'network_type': 'mobile0.25',
          'confidence_threshold': 0.02,
          'top_k': 5000,
          'nms_threshold': 0.4,
          'keep_top_k': 750,
          'vis_threshold': 0.5,
          'threshold': 1.15}
