## Project Description

For this personal project, I put together all of the work I have done for detection of masks in images and facial recognition in images to create a larger application that can identify an individual in an image regardless of them wearing a mask or not. The main idea here is that the system is able to recognize one person in an image, where the person to be recognized is specified by the user, when the person is either wearing a face mask or not wearing a face mask. If the system recognizes this person in the image then it will label the outcome of the process as "Recognized". If the system does not recognize the individual in frame, regardless of them wearing a face mask or not, then it will label the ourcome of the process as "Not Recognized". 

I belive this application has many use cases given that most poeple are wearing a mask due to the pandemic and that facial recognition software is increasing in use in many industries. If scaled correctly, this application or similar ones will be very valuable in facial recognition systems that must ID individuals regardless of them wearing a mask or not. With that said, my personal motivation for creating this software comes from the fact that one day I got extremely annoyed that my iPhone's Face ID didn't recognize me when I was wearing a face mask. My annoyance pushed me to create this system and achieve reasonable validation scores.

There are some limitations that I had to accept when implementing this system, for the sake of finishing on time. These are:

- Only one individual can be recognized with this existing system (it is possible to scale it up)

- This sytem can get buggy when presented with multiple faces in an image, so it's best to keep to 1 person

With that said, I did implemented three interfaces for interacting with the system, which are:

1) **Live Feed Processing**: The system will start reading in frames from a live feed such as a security camera or a simple desktop cam, and perform real-time classification on the feed.

2) **Single Image Processing**: The system will read in only one image and perform classification directly on the inputted image.

3) **Batch Image Processing**: The system will read in a batch of images from an inputted directory and will classify all of the images at once.


## System Design

The overall system is best described by the following diagram that I created:

![](/Diagrams/System%20Diagrams-Entire%20System%20Diagram.jpg)

The three interfaces, as described before, are labeled at the very top of the diagram. Each interface feeds the image(s) or frame(s) into the **Image Processing Pipeline** which acts as an intermediary between the interfaces and the classifiers. This pipeline script processes only one image at a time so the live feed interface and batch image interface must feed the images needing to be processed in a sequential order into the pipeline. Moreover, the pipeline feeds the single image into the mask detection classifier and in return recieves a signal telling the pipeline if the detected face in the image is wearing a face mask or not. Given this signal, the Image Processing Pipeline then calls the correct **Facial Recognition Identifier** either the one trained on masked individuals (**Masked Individuals**) or the one trained on unmasked individuals (**Unmasked Individuals**). Each Facial Identifier then classifies the face in the image as "Recognized" or "Not Recognized" which is converted into a signal and sent back to the pipeline. 

Let's now take a closer look at each of the three classifiers.

`Mask Detection Classifier`: This classifier identifies if the person(s) in an inputted image is wearing a face mask or not wearing a face mask. I primarily used OpenCV to process the dataset of images, and TensorFlow for actually training and testing the models. I considered a lightweight Convolutional Neural Network structure called MobileNet v2 as my Deep Learning model. I then tailored the model for binary classification (Mask or No Mask) by using a Sigmoid function to configure my output. After 5 epochs, I managed to acheive a 98% validation accuracy on test images. For better analysis of the person's face in the image, I used Haar Cascades to first extract the frontal faces region located in the frame, and then fed into my model for prediction. The overall workflow of building and using this classifier can been seen in the diagram below:

![](/Diagrams/System%20Diagrams-Mask%20Detector.jpg)

`Unmasked Identifier`: This classifier is able to classify if the person in the inputted image, who is not wearing a face mask, is recognized or not recognized. Once again, the user of the system defines the person that is supposed to be recognized by the system by training the classifier on images of that person. This classifier also uses the MobileNet v2 structure and is also tailored for binary classification (Recognized or Not Recognized). For improved results I also used Haar Cascades to first extract the frontal face region of the image and then perform classification. The overall workflow of building and using this classifier can been seen in the diagram below:

![](/Diagrams/System%20Diagrams-Unmasked%20Identifier.jpg)

`Masked Identifier`: This classifier is able to classify if the person in the inputted image, who is wearing a face mask, is recognized or not recognized. Once again, the user of the system defines the person that is supposed to be recognized by the system by training the classifier on images of that person. This classifier also uses the MobileNet v2 structure and is also tailored for binary classification (Recognized or Not Recognized). I tested Haar Cascades with this system but actually saw that the classifier performs worst when the frontal face region is first extracted in comparison to no extraction. I mostly attributed this to Haar Cascades having a difficult time detecting the frontal face region of the image given that half the face is covered with a mask. So I actually didn't use Haar Cascades with this classifier. The overall workflow of building and using this classifier can been seen in the diagram below:

![](/Diagrams/System%20Diagrams-Masked%20Identifier.jpg)

## Testing the System

I trained the facial identification classifiers on my own face, masked and unmasked, and tested it accordingly. Note that this system can be trained to recognize any one persons face so there is no restrictions on only recognizing my face. Moreover, I used Batch Image Processing to processes several test images (images it hasn't seen before) and see the outcome. Although not perfect, the classifier did manage to do pretty well. See below for the outcome of the Batch Image processing.

![](/Output/Not_Recognized!_1609964614.5133271.png)

![](/Output/Recognized!_1609964627.393876.png)

![](/Output/Not_Recognized!_1609964638.096912.png)

![](/Output/Recognized!_1609964618.2151308.png)

More test examples can be seen the output directory of the repo.

## Building the Project
