## Description: This script assumes that the face in the image is masked and then tries to classify as a face it recognizes
##				or it doesn't recognize. Obviously the recognition is defined by the user of this application in the training phase of this model.


## Load packages and dependancies ##
import os
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow import keras
from tensorflow.keras import layers
import cv2
import numpy as np
import time


## Utility Functions

# Function for applying transformations necessary transformations to the image
def apply_image_trans(img):

	# Coverting to image size that requires (224 x 224)
	img_size = 224

	# apply further transformations
	final_image = cv2.resize(img, (img_size, img_size))
	final_image = np.expand_dims(final_image, axis = 0)
	final_image = final_image/255.0
	
	return final_image


# Function for interpreting the sigmoid output
# For sake of simplicity, we aren't accounting for a sigmoid = 0.50
def convert_sigmoid_output(sigmoid):

	if(sigmoid < 0.50):
		print('-> This person is Recognized!\n')
		return 1
	elif(sigmoid > 0.50):
		print('-> This person is not Recognized!\n')
		return 0


## Main Function to be Called ##

# Recognize the face within the image
def identify_masked(img):

	# Read in the trained model
	model = keras.models.load_model('./Person_With_Mask_Detection.h5')

	# obtain transformed image
	final_image = apply_image_trans(img)

	# Run the transformed image through the classifier
	Prediction = model.predict(final_image)       

	# Set the status tag
	if(Prediction > 0.50):
		status = "Not_Recognized!"   
	elif(Prediction < 0.50):
		status = "Recognized!"
	else:
		status = "Not Sure!"


	# Add text to the raw inputted image
	font_scale = 5
	font = cv2.FONT_HERSHEY_PLAIN
	cv2.putText(img, status, (0, 50), font, font_scale, (255,0,0), 4)

	# Save the image to disk for further investigation (if not required just comment this out)
	date_string = str(time.time())
	pic_name = './Output/' + status + '_' + date_string + '.png'
	cv2.imwrite(pic_name, img)

	return(convert_sigmoid_output(Prediction)) 
