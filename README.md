# Face Recognition
Real-time face recognition in unconstrained environments. Using [Shuffle Attention MobileNetV3](https://github.com/SajjadAemmi/SA-MobileNetV3) Architecture as Backbone, and modified ArcFace as loss function.

This module can get a number of names as input for tracking the specific face.

![model arch](assets/output.jpg)

## Experiments

#### on LFW

Attempt | Parameters | Madds | Top1-acc
--- | --- | --- | --- |
MobileNetV3-Large |  |  |  | 
SA-MobileNetV3-Large |  |  |  |
SA-MobileNetV3-Large with modified ArcFace loss |  |  |  |

## Installation
1- clone the repository

2- install requirements
```
pip install -r requirements.txt
```
3- download required model files: 
```
python download_weights.py
```


## Train
Put your dataset in `./datasets` directory.
Run the following command for train model on your own dataset:
```
python dataset_utils/preprocess_dataset.py --dataset-path ./datasets/CelebA 
```

In root directory of project, run the following command: 

Run the following command for train model on your own dataset:
```
python train.py --dataset mnist 
```

## Test

Run the following command for evaluation trained model on test dataset:
```
python test.py --dataset mnist
```

## Predict

Run the following command for classification images:
```
python predict.py --input /path/to/image.jpg 
```


## Inference
Put your input images or videos in ./input directory. The output will be saved in ./output. 
In root directory of project, run the following command: 
```
python inference_video.py --input "./input/sample.mp4" -u
```
or
```
python inference_image.py --input "./input/sajjad.jpg" -u
```

Use -sh for representation of results during code running or not

Note that you can pass some other arguments. Take a look at *main.py* file.