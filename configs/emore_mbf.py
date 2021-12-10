from easydict import EasyDict as edict

# make training faster
# our RAM is 256G
# mount -t tmpfs -o size=140G  tmpfs /train_tmp

config = edict()
config.loss = "arcface"
config.network = "mbf"
config.resume = False
config.output = None
config.embedding_size = 512
config.sample_rate = 1.0
config.fp16 = True
config.momentum = 0.9
config.weight_decay = 2e-4
config.batch_size = 256
config.lr = 0.1  # batch size is 512

config.rec = "datasets/faces_emore"
config.num_classes = 85742
config.num_image = 5822653
config.num_epoch = 16
config.warmup_epoch = -1
config.decay_epoch = [8, 14, ]
config.val_targets = ["lfw", ]