# Write a Python Script that captures images from your webcam video stream
# Extracts all Faces from the image frame (using haarcascades)
# Stores the Face information into numpy arrays

# 1. Read and show video stream, capture images
# 2. Detect Faces and show bounding box (haarcascade)
# 3. Flatten the largest face image(gray scale) and save in a numpy array
# 4. Repeat the above for multiple people to generate training data

import cv2
import numpy as np

# Initialize Camera
cap = cv2.VideoCapture(0)

# Face Detection
# Load Harcascade file_namee
face_cascade = cv2.CascadeClassifier("/home/yuvraj/Downloads/hid/x/Coding Blocks ML/17-Project - Real Time Face Recognition using KNN/haarcascade_frontalface_alt.xml")

skip = 0
face_data = []
dataset_path = '/home/yuvraj/Downloads/hid/x/Coding Blocks ML/17-Project - Real Time Face Recognition using KNN/facedata/' # Store the values in this folder
file_name = input("Enter the name of the person:") # whose face we are scanning

while True:
	ret, frame = cap.read()

	if ret == False:
		continue

	gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # RGB frame to Gray frame


	cv2.imshow("Frame",frame)
	faces = face_cascade.detectMultiScale(frame,1.3,5)
	# print(faces) # To print the coordinates of the detected faces
	faces = sorted(faces,key = lambda f:f[2]*f[3]) # Store the faces in sorted order


	for face in faces[-1:]: # pick up the large face (start from largest face i.e., last face)
		x,y,w,h = face
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

		# Extract (Crop out the required face) : Region of Interest
		offset = 10 # give a padding of 10px to all sides
		face_section = frame[y-offset:y+h+offset,x-offset:x+w+offset]
		face_section = cv2.resize(face_section,(100,100)) # resize to 100*100 image

		# Store every 10th face not every face
		skip += 1		
		if skip%10==0:
			face_data.append(face_section)
			print(len(face_data)) # How many faces i have captured so far



	cv2.imshow("Frame",frame)
	cv2.imshow("Face Section",face_section)


	key_pressed = cv2.waitKey(1) & 0xFF
	if key_pressed == ord('q'):
		break

# Convert our face list array into a numpy array
face_data = np.asarray(face_data)
face_data = face_data.reshape((face_data.shape[0],-1))
print(face_data.shape)

# Save this data into file system
np.save(dataset_path+file_name+'.npy',face_data)
print("Data Successfully Saved at "+dataset_path+file_name+'.npy')

cap.release()
cv2.destroyAllWindows()
