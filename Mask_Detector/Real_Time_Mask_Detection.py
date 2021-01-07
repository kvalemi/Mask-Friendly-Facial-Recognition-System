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



## Utility Functions ##

# Function for applying transformations
def apply_image_trans(img):
    final_image = cv2.resize(img, (img_size, img_size))
    final_image = np.expand_dims(final_image, axis = 0)
    final_image = final_image/255.0
    
    return final_image

# Function for interpreting the sigmoid output
def convert_sigmoid_output(sigmoid):
    if(sigmoid < 0.50):
        print('Mask Detected!')
    elif(sigmoid > 0.50):
        print('Mask not Detected!')
    else:
        print('Not Sure!')



## Starting Point ##

# Upload the model (if you don't want to train it)
new_model = keras.models.load_model('../Mask_detection_Model.h5')

## some variables we will need for the Web Cam detection

# font size of detection box
font_scale = 3

# font type of detection text
font = cv2.FONT_HERSHEY_PLAIN

# path to haar cascades model
path = '../haarcascade_face.xml'

# Coverting to image size that is required by ImageNet (224 x 224)
img_size = 224

# Start Webcam realtime Recording
capture = cv2.VideoCapture(0)


## Start parsing the frames in real time ##
while True:
    # capture the frame
    ret, img = capture.read()
        
    # focus on face inside of the frame
    faceCascade = cv2.CascadeClassifier(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1, 7)

    # loop through the face in the picture
    for x,y,w,h in faces:
        
        # Apply transformations
        roi_color = img[y:y+h, x:x+w]

        # obtain transformed face
        final_image = apply_image_trans(roi_color)

        # this is the output of the model
        Prediction = new_model.predict(final_image)
        convert_sigmoid_output(Prediction)        

        if(Prediction > 0.50):

            # Set No Mask status
            status = "No Mask"   

            # Add the text
            cv2.putText(img, status, (x, y), font, font_scale, (0,0,255), 2)

            # Add the rectangle
            cv2.rectangle(img, (x,y), (x + w, y + h), (0,0,255), 2)

        else:

            # Set No Mask status
            status = "Face Mask"

            # Add the text
            cv2.putText(img, status, (x, y), font, font_scale, (255,0,0), 2)

            # Add the rectangle
            cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)            


    # show image
    cv2.imshow('Face Mask Detection', img)
    
    if cv2.waitKey(0):
        break

            
# release the capture        
capture.release() 
cv2.destroyAllWindows()