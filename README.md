# Introduction

In this project, I implemented an action recognition system which can recognize 5 types of human action: kick, punch, stand, wave, squat.

# Installation

check the: settings/library.txt

#Run real-time action recognition

python3 src/run_detector.py --images_source webcam

# mplementation

Have 5 types of action: kick, punch, stand, wave, squat. The length of each video about 2 minutes and each video only contains one type of action with a frame size of 640x480, and then saved to images.

# Get Skeleton from image

I used OpenPose to detect the human pose in each training images.

The output skeleton format of OpenPose can be found at OpenPose Demo - Output.

The generated training data files are located in data folder:

+ skeleton_raw.csv: original data

+ skeleton_filtered.csv: filtered data where incomplete poses are eliminated

# Data processing

1. Remove all joints on head: because all joint positions on head are not required for action recognition task.

2. Normalization: all joint positions are converted to the x-y coordinates relative to the skeleton bounding box.

# Deep learning model

I built a neural network by using Keras. the model consists of three hidden layers and a softmax output layer to conduct a 5-class classification.

The model implemented in action_training.ipynb.

The generated model is saved in model folder.
