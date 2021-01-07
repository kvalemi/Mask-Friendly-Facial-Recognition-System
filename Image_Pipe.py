## Description: This function specified in this module takes as input an image and first feeds into the mask classifier. If a mask is detected in the image
##				then the Face Classifier that aacounts for masks is called, otherwise, if no mask is detected then the non-mask
#				Face Classifier is called.


## Load the dependancies ##
import cv2
import numpy as np
from Detect_Mask import detect_mask
from Identify_Masked_Face import identify_masked
from Identify_Unmasked_Face import identify_non_masked

## Main Function to Be Called ##

def classify_image(img):

	# call mask detector classifier
	if(detect_mask(img)):
		# Face classifier that accounts for masks
		ret = identify_masked(img)
	else:
		# Face classifier that doesn't need to account for masks
		ret = identify_non_masked(img)

	return ret
