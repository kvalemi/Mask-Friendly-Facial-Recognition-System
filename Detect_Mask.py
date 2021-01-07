## Description: This classifier detects if the faces located in an image are wearing face masks or not. This is the first clasifier
##				in the overall image processing pipeline.


## Load packages and dependancies ##
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) # mute tensorflow warnings
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np


## Utility Functions ##

# Function for applying transformations necessary transformations to the image
def apply_image_trans(img):

	# Coverting to image size that requires (224 x 224)
	img_size = 224

	# Use Haar Cascades object classifier to crop everything but the face out of the image
	faceCascade = cv2.CascadeClassifier('./haarcascade_face.xml')
	# Haar cascades requires the raw image to be converted to gray scale
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	faces = faceCascade.detectMultiScale(gray, 1.1, 5)

	# find location of the face
	x = faces[0][0]
	y = faces[0][1]
	w = faces[0][2]
	h = faces[0][3]

	# crop the face
	final_img = img[y:y+h, x:x+w]

	# apply further transformations
	final_image = cv2.resize(final_img, (img_size, img_size))
	final_image = np.expand_dims(final_image, axis = 0)
	final_image = final_image/255.0
	
	return final_image


# Function for interpreting the sigmoid output
# For sake of simplicity, we aren't accounting for a sigmoid = 0.50
def convert_sigmoid_output(sigmoid):

	if(sigmoid < 0.50):
		print('-> Mask Detected!')
		return 1
	elif(sigmoid > 0.50):
		print('-> Mask not Detected!')
		return 0


## Main Function to be Called ##

# Function to begin image processing and further classification
# to see if there is a mask in the faces of the image
def detect_mask(img):

	# Read in the trained model
	new_model = keras.models.load_model('./Mask_detection_Model.h5')

	# obtain transformed face in the image
	final_image = apply_image_trans(img)

	# detect masks
	Prediction = new_model.predict(final_image)
	
	return(convert_sigmoid_output(Prediction))        
