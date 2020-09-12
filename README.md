# Face Recognition
Real-time face recognition in unconstrained environments

This module can get a number of names as input for tracking the specific face.

## Installation

1. clone the repository
2. download required model files: `python downloadmodels.py`

## Test and run
Put all of your desired input videos in ./video directory. The output will be saved in ./result. 
In root directory of project, run the following command: 
```
python main.py -i "./video/sample.mp4" -u  --type name  -names ['Obama','Joaquin','Connie']
```
Use -sh for representation of results during code running or not

Note that you can pass some other arguments. Take a look at *main.py*, *parse_args* fanction.