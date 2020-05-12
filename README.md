# Project-6-Real-Time-Face-Recognition-using-OpenCV-and-knn
## Introduction:
The face is the most crucial entity for human identity. It is the feature that best distinguishes a person. And for the very same reasons, Face Recognition is an important technique. Face recognition is an interesting and challenging problem and impacts important applications in many areas such as identification for law enforcement, authentication for banking and security system access, and personal identification among others.

Face recognition is an easy task for humans but its an entirely different task for a computer. A very little is known about human recognition to date on How do we analyze an image and how does the brain encode it and Are inner features (eyes, nose, mouth) or outer features (head shape, hairline) used for successful face recognition? Neurophysiologist David Hubel and Torsten Wiesel have shown that our brain has specialized nerve cells responding to specific local features of a scene, such as lines, edges, angles or movement. Since we don‟t see the world as scattered pieces, our visual cortex must somehow combine the different sources of information into useful patterns. Automatic face recognition is all about extracting those meaningful features from an image, putting them into a useful representation and performing some classifications on them.

The whole process can be divided into three major steps where the first step is to find a good database of faces with multiple images for each individual. The next step is to detect faces in the database images and use them to train the face recognizer and the last step is to test the face recognizer to recognize faces it was trained for.
![](https://i.imgur.com/EHABvUI.png)

## Problem Statement:
Face recognition is an easy task for humans but its an entirely different task for a computer. So our task is to
recognize the faces of differnt persons and build a pretty box around different faces to detect the faces.


## Requirements:
- Python
- PyCharm or any IDE to run python file
- OpenCV

## Haar Cascade Dataset:
Haar Cascade is a machine learning object detection algorithm used to identify objects in an image or video and based on the concept of ​​ features proposed by Paul Viola and Michael Jones in their paper "Rapid Object Detection using a Boosted Cascade of Simple Features" in 2001.

It is a machine learning based approach where a cascade function is trained from a lot of positive and negative images. It is then used to detect objects in other images.

The algorithm has four stages:

  1. Haar Feature Selection
  2. Creating  Integral Images
  3. Adaboost Training
  4. Cascading Classifiers
It is well known for being able to detect faces and body parts in an image, but can be trained to identify almost any object.

![](https://lh3.googleusercontent.com/proxy/TgjS4w4vJpoBop6rAHz9cz-nwgdB1u5f3QUTb9d_Xowz7tdSsPt-f7VSmuvz3KBiEea31Vjcz3DJya8op-SeZulzCIzKYFw2SkIn0wy2cEYuhuvy8f6nV4eVB_ENhqDqj0D5whYNZw)

## Technology Stack:
- Python — The whole code has been written in Python
- cv2 — cv2 is the OpenCV module and is used here for reading & writing images & also to input a video stream
- Algorithm — KNN
- Classifier — Haar Cascades

## Working/Implementation:
`Step-1==>` **Generating Training Data: The following steps are followed to generate training data**

- Write a Python Script that captures images from your webcam video stream.
- Extracts all Faces from the image frame (using haar cascades).
- Stores the Face information into numpy arrays.

    1. Read and show video stream, capture images.
    2. Detect Faces and show bounding box (haar cascade).
    3. Flatten the largest face image(gray scale) and save in a numpy array.
    4. Repeat the above for multiple people to generate training data.
    
`Step-2==>`**Building The Face Classifier**
- Recognise Faces using the classification algorithm — KNN
    
    1. load the training data (numpy arrays of all the persons).
    
    x-values are stored in the numpy arrays
    
    y-values we need to assign for each person
    
    2. Read a video stream using opencv.
    3. extract faces out of it.
    4. use knn to find the prediction of face (int).
    5. map the predicted id to name of the user.
    6. Display the predictions on the screen — bounding box and name.


## KNeighborsClassifier (KNN):
In pattern recognition, the k-nearest neighbors algorithm (k-NN) is a non-parametric method used for classification and regression. In both cases, the input consists of the k closest training examples in the feature space. The output depends on whether k-NN is used for classification or regression:

In k-NN classification, the output is a class membership. An object is classified by a plurality vote of its neighbors, with the object being assigned to the class most common among its k nearest neighbors (k is a positive integer, typically small). If k = 1, then the object is simply assigned to the class of that single nearest neighbor.
In k-NN regression, the output is the property value for the object. This value is the average of the values of k nearest neighbors.
The k-nearest neighbors classification algorithm is implemented in the KNeighborsClassifier class in the sklearn.neighbors module.

![](https://images.squarespace-cdn.com/content/v1/55ff6aece4b0ad2d251b3fee/1465017787823-KXFG6O0MU5NWYF8EI6UU/ke17ZwdGBToddI8pDm48kICIavOU0GBCWw19s1p5lSVZw-zPPgdn4jUwVcJE1ZvWULTKcsloFGhpbD8VGAmRSUJFbgE-7XRK3dMEBRBhUpycqPLetyMM_eWnzi1H9kYzvMtuY8jA9E1WuBOqLarM1WXLSloz6LILkqH1WHTAqb8/image-asset.png)
    
## Pros and Cons:
Among the different biometric techniques, facial recognition may not be the most reliable and efficient. However, one key advantage is that it does not require the cooperation of the test subject to work. Properly designed systems installed in airports, multiplexes, and other public places can identify individuals among the crowd, without passers-by even being aware of the system. Other biometrics like fingerprints, iris scans, and speech recognition cannot perform this kind of mass identification. However, questions have been raised on the effectiveness of facial recognition software in cases of railway and airport security.

## Summary:
Face recognition is a crucial security application. Through this project, a very basic form of face recognition has been implemented using the Haar Cascades Classifier, openCV & K-Nearest Neighbors Algorithm.
