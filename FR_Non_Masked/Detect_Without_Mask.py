## Load packages and dependancies ##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import pickle
import sys



## Utility Functions

# Function for applying transformations
def apply_image_trans(img):
    img_size = 224
    faceCascade = cv2.CascadeClassifier('../haarcascade_face.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 5)                    

    x = faces[0][0]
    y = faces[0][1]
    w = faces[0][2]
    h = faces[0][3]
    roi_color = img[y:y+h, x:x+w]    
    
    final_image = cv2.resize(roi_color, (img_size, img_size))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0
    
    return final_image

# Function for interpreting the sigmoid output
def convert_sigmoid_output(sigmoid):
    if(sigmoid < 0.50):
        print('\nThis person is Recognized!\n')
    elif(sigmoid > 0.50):
        print('\nThis person is not Recognized!\n')
    else:
        print('\nNot Sure!\n')


## Starting Point ##

# Upload the model (if you don't want to train it)
model = keras.models.load_model('../Person_Without_Mask_Detection.h5')

## some variables we will need for the Web Cam detection

# font size of detection box
font_scale = 5

# font type of detection text
font = cv2.FONT_HERSHEY_PLAIN

# Coverting to image size that is required by ImageNet (224 x 224)
img_size = 224

# input image path
img_path = sys.argv[1]

# Load image
img = cv2.imread(img_path)

# obtain transformed face
final_image = apply_image_trans(img)

# this is the output of the model
Prediction = model.predict(final_image)
convert_sigmoid_output(Prediction)        