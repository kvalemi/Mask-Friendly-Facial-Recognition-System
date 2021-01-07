## Description: This script processes a a live feed of by reading in frames from a specified webcam with OpenCV functions and
##              parsing the frames of the live feed with the image processing pipeline.


## Load in the dependancies ##
import cv2
import sys
import os
from Image_Pipe import classify_image


## Starting Point ##

def main():

    print('Beginning Live Feed Processing\n.\n.\n.\n')
    
    # Define live feed camera
    capture = cv2.VideoCapture(0)

    # Begin processing the frames of the live feed
    while True:

        try:

            # capture the frame
            ret, img = capture.read()

            # call the image classifier
            classify_image(img)

        except Exception as e:
            # exception thrown case
            print('** Something Failed **')
            break

        # break out of loop when user presses 0
        if cv2.waitKey(0):
            break    

    print('\n.\n.\n.\nLive Feed Processing Finished\n')

    
    # release the capture        
    capture.release() 
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()