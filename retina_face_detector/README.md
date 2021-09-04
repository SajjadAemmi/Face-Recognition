# Retina Face Detector
Real-time face detection in unconstrained environments. 

![model arch](./output/test.jpg)


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


## Inference
Put your input images or videos in ./input directory. The output will be saved in ./output. 
In root directory of project, run the following command for image: 

```
python inference_image.py --input "./input/test.jpg"
```
and for video:
```
python inference_video.py --input "./input/obama.mp4"
```
Use -sh for representation of results during code running or not

Note that you can pass some other arguments. Take a look at *inference_video.py* file.


## Train


## Test


## Predict


