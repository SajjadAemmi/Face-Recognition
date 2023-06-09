# Face Recognition with Arcface

Real-time face recognition in unconstrained environments, based on [InsightFace](https://github.com/deepinsight/insightface). 

## Installation

install requirements
```
pip install -r requirements.txt
```

## Inference

For face detection, run the following command:
```
python inference_detection.py --input input/sajjad.jpg
```

For face identification, feature extraction and compare, run the following command:
```
python inference_identification.py --input1 input/sajjad0.jpg --input2 input/sajjad1.jpg
```

For face recognition, run the following command:
```
python inference_recognition.py --input ./io/input/sajjad2.jpg --update
```
The output will be saved in `./output` by default.

If you want recognize your own faces, first you should put your images in `./face_bank` directory with your names as directories name in this structure:
```
face_bank
│
└───Ali
│   │   ali_image_1.jpg
│   │   ali_image_2.jpg
│   │   ...
│   
└───Sara
│   │   sara_image_1.jpg
│   │   sara_image_2.jpg
│   │   ...
│
│
...
```
Then run same command as above with `--update` argument. Note that After each change in `./face_bank` directory, you should use `--update` again.
