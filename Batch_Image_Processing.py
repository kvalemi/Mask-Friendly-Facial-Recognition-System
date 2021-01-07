## Description: This script processes a batch of images image by taking as input the relative or absolute path
##				of the directory of images and parses it using the image processing pipeline.


## Load in the dependancies ##
import cv2
import sys
import os
from Image_Pipe import classify_image


## Starting Point ##

def main():

	print('Beginning Batch Image Processing\n.\n.\n.\n')

	# read in the batch directory
	img_dir = sys.argv[1]

	# Start batch processing images in the directory
	for filename in os.listdir(img_dir):

		try:

			print('---')

			# obtain the relative path of the current file
			rel_path = os.path.join(img_dir, filename)

			print('\n-> Processing: ' + str(rel_path) + '\n')

			# read in the image
			img = cv2.imread(rel_path)
			# call the image classifier
			classify_image(img)

			print('---')

		except Exception as e:
			# exception thrown case
			print('** Something Failed @ ' + str(filename) + ' **')
			pass

	print('\n.\n.\n.\nBatch Image Processing Finished.\n')


if __name__ == "__main__":
	main()