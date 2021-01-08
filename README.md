## Project Description

In this personal project, I put together all of the work I've done for deep learning, computer vision, and system design to create a large stand-alone application that can recognize an individual in an image regardless of them wearing a mask or not. The central idea is that the system can recognize one person in an image, where the person to be recognized is specified by the user of the system, when the person in the image is either wearing a face mask or not wearing a face mask. If the system recognizes this person then it will label the outcome of the process as "Recognized". If the system does not recognize the individual in the frame, regardless of them wearing a face mask or not, then it will label the outcome of the process as "Not Recognized". 

I believe this application has many use cases given that most people nowadays are wearing a mask due to the pandemic, and also the fact that facial recognition systems are increasing in popularity in many industries. If scaled correctly, this application could be very valuable in facial recognition systems that must ID individuals regardless of them wearing a mask or not. This use-case can be implemented in a security system at an office where the individual trying to enter the premises of the office must first be identified as a valid employee either wearing a mask or not. With all of this said, the personal motivation for creating this application comes from the fact that one day I got annoyed that my iPhone's Face ID didn't recognize me when I was wearing a face mask and this annoyance pushed me to create a system that did recognize me regardless of wearing a mask.

There are some limitations that I had to accept when implementing this system, for the sake of finishing on time. These are:

- Only one individual can be recognized with this existing system (it is possible to scale it up with minor changes)

- This system can get buggy when presented with multiple faces in an image, so it's best to keep to 1 person

With that said, I did implement three interfaces for interacting with the system, which are the following:

1) **Live Feed Processing**: The system will start reading in frames from a live feed such as a security camera or a desktop webcam, and perform real-time classification on the images of the feed.

2) **Single Image Processing**: The system will read in only one image and perform classification directly on the sole inputted image.

3) **Batch Image Processing**: The system will read in a batch of images from an inputted directory and will classify all of the images at once.



## System Design

The overall system is best described by the following diagram that I created:

![](/Diagrams/System%20Diagrams-Entire%20System%20Diagram.jpg)

The three interfaces, as described before, are labelled at the very top of the diagram. Each interface feeds the image(s) or frame(s) into the **Image Processing Pipeline** which acts as an intermediary between the interfaces and the classifiers. This image pipeline processes only one image at a time so the live feed interface and batch image interface must feed the images in a sequential order into the pipeline. Moreover, the pipeline feeds the single image into the **Mask Detection Classifier** and in return receives a signal informing the pipeline if a face mask was detected or not on the frontal face region of the image. Given this signal, the Image Processing Pipeline then calls the correct **Facial Recognition Identifier**, being either the one trained on masked individuals (**Masked Identifier**) or the one trained on unmasked individuals (**Unmasked Identifier**). Each Facial Identifier then classifies the face in the image as "Recognized" or "Not Recognized" which is then converted into a signal and sent back to the pipeline. 

Let's now take a closer look at each of the three classifiers.

`Mask Detection Classifier`: This classifier identifies if the person in an inputted image is wearing a face mask or not. I primarily used OpenCV to process the dataset of images, and TensorFlow/Keras for actually training and testing the models. I considered a lightweight Convolutional Neural Network structure called MobileNet v2 as my Deep Learning model. I then tailored the model for binary classification (Mask or No Mask) by using a Sigmoid function to configure my output correctly. After 5 epochs, I managed to achieve a 98% validation accuracy on test images. Moreover, to improve the classification I used Haar Cascades to first extract the frontal face region of the frame and then fed it into my model for prediction. The overall workflow of building and using this classifier can be seen in the following diagram:

![](/Diagrams/System%20Diagrams-Mask%20Detector.jpg)

`Unmasked Identifier`: This classifier can classify the person in the inputted image, who is not wearing a face mask, is recognizable or not recognizable. Once again, the user of the system defines the person that is supposed to be recognized by training the classifier on unmasked images of that person. This classifier also uses the MobileNet v2 structure and is also tailored for binary classification (Recognized or Not Recognized). To improve the classification I  used Haar Cascades to first extract the frontal face region of the image which is then fed into my model. The overall workflow of building and using this classifier can be seen in the following diagram:

![](/Diagrams/System%20Diagrams-Unmasked%20Identifier.jpg)

`Masked Identifier`: This classifier can classify the person in the inputted image, who is wearing a face mask, is recognizable or not recognizable. Once again, the user of the system defines the person that is supposed to be recognized by training the classifier on masked images of that person. This classifier also uses the MobileNet v2 structure and is also tailored for binary classification (Recognized or Not Recognized). I tested Haar Cascades with this classifier but saw that the classifier performed worst when the frontal face region is extracted in comparison to no extraction. I believe this is due to Haar Cascades having a difficult time detecting the frontal face region of the image given that half the face is covered by a mask. So I didn't use Haar Cascades with this classifier. The overall workflow of building and using this classifier can be seen in the diagram below:

![](/Diagrams/System%20Diagrams-Masked%20Identifier.jpg)



## Testing the System

I trained all of the facial identification classifiers on my face, masked and unmasked. Note that this system can be trained to recognize any one person's face so there are no restrictions on only recognizing my face with this system. Moreover, I used Batch Image Processing to processes several test images (images it hasn't seen before). Although not perfect, the classifier did manage to do pretty well. See below for the outcome of some of the test images.

![](/Output/Not_Recognized!_1609964614.5133271.png)

![](/Output/Recognized!_1609964627.393876.png)

![](/Output/Not_Recognized!_1609964638.096912.png)

![](/Output/Recognized!_1609964618.2151308.png)

More test examples can be seen in the output directory of the repo (`./output`).



## Building the Project

1) Populate the `./FR_Masked/Dataset/Person/` directory with masked pictures of the person wanting to be recognized by the system. Examples of these types of pictures can be seen in the `./Test_Images/` directory (only consider masked pictures of my face from this directory).

2) Populate the `./FR_Non_Masked/Dataset/Person/` directory with non-masked pictures of the person wanting to be recognized by the system. Examples of these types of pictures can be seen in the `./Test_Images/` directory (only consider unmasked pictures of my face from this directory).

3) Populate the `./Mask_Detector/Dataset/Face_Mask/` directory with images of one person wearing a mask.

4) Populate the `./Mask_Detector/Dataset/No_Mask/` directory with images of one person not wearing a mask.

(note for steps 3 and 4 you can use this dataset from Kaggle: https://www.kaggle.com/andrewmvd/face-mask-detection)

5) Run the following three training scripts to train each of the classifiers:

- `./FR_Masked/Train_Model.py`

- `./FR_Non_Masked/Train_Model.py`

- `./Mask_Detector/Train_Model.py`

A notebook is located in each of the classifiers directories in case you would like to take a closer look at the training code inside of a notebook.

6) Once all of the models are serialized to disk as `h5` files, then you can execute each of the interfaces like so:

- `Python3 Single_Image_Processing.py '[directory of single image]'`

- `Python3 Batch_Image_Processing.py '[directory or batch of images]'`

- `Python3 Live_Feed_Processing.py` (Ensure that the identification code of the camera is set up correctly in the script)

