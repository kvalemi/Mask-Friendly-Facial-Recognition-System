## Description: This script processes a single image by taking as input the relative or absolute directory
##				of the image and parsing it using the image processing pipeline.


## Load in the dependancies ##
import cv2
import sys
import os
from Image_Pipe import classify_image


## Starting Point ##

def main():

	print('Beginning Single Image Processing\n.\n.\n.\n')

	# begin single image processing
	try:

		print('---')

		# read in the directory
		img_path = sys.argv[1]

		print('-> Processing: ' + str(img_path))

		# read in the image
		img = cv2.imread(img_path)
		# call the classifier
		classify_image(img)

		print('---')

	except Exception as e:
		# exception thrown case
		print('** Something Failed **')
		pass

	print('\n.\n.\n.\nSingle Image Processing Finished.\n')


if __name__ == "__main__":
	main()