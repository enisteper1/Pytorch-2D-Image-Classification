# Pytorch-2D-Image-Classification

# Introduction

This repository makes the user able to explore and test the custom created pytorch model.

# Training

At the beginning you need to put your seperated class folders in `data/images` directory like I put cats and dogs folders.
Then you need to write those class names in <data/classes.txt>

**For the first time with new dataset** `python3 train.py --new_data`

**To resume training**	`python3 train.py --resume`

**To visualize the improvement of the model** `python3 visualize.py`

Depending on the your dataset size and classes you can vary the convolutional or linear layers in `model.py` to make model more accurate.

# Detection

<p>Put your input images in `data/inputs` folder then just run `python3 detection.py`</p>
<img src="https://user-images.githubusercontent.com/45767042/101196301-9b525280-3671-11eb-9d07-0713881aaaea.jpg">
<p>You can get the outputs from `data/outputs` folder.</p>
<img src="https://user-images.githubusercontent.com/45767042/101196362-b1f8a980-3671-11eb-8a71-9debdd308bb8.jpg">

# Reference

Main references were https://www.ultralytics.com. and https://pythonprogramming.net/ 
My general purpose was combining their projects with the aim of making this type of project reachable and modifiable at ease.
