""" 
Python script for detecting and blurring faces
"""
from pathlib import Path
from matplotlib import pyplot as plt
#import OpenCV library
import cv2

# test: reading an image
#img = Image.open('photo/celeb1.png')
#img.show()

"""  
Convert to gray for opencv bc it expects gray images
Source: https://www.superdatascience.com/blogs/opencv-face-detection
"""
def convert_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# loading up cascade classifier training file for haarcascade
haar_face_cascade = cv2.CascadeClassifier('data/haarcascade_frontalface_alt.xml')

"""  
The code is pretty much the same as before; it’s just grouped inside a 
function for reusability detect faces
"""
def detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = colored_img.copy()

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    print('Faces found: ', len(faces)) # give the number of faces

    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
    return img_copy

#blur effect
def blur_detect_faces(f_cascade, colored_img, scaleFactor = 1.1):
    img_copy = colored_img.copy()

    gray = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)

    faces = f_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=5)
    #print('Faces found: ', len(faces))

    #https://www.geeksforgeeks.org/python/how-to-blur-faces-in-images-using-opencv-in-python/
    for (x, y, w, h) in faces:
        cv2.rectangle(img_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #blur effect
        roi = img_copy[y:y+h, x:x+w]
        roi = cv2.GaussianBlur(roi, (75, 75), 30) # here to change the intensity of the blur
        img_copy[y:y+h, x:x+w] = roi
    return img_copy

""" Cours de traitment d'image
TP 7 
Line 16
"""
INPUT_DIR = 'photo'
# for loop in the png images in the photo folder
for file_name in Path(INPUT_DIR).iterdir():
    if file_name.suffix == ".png":
        img = cv2.imread(str(file_name))
       # blurring of the face
        faces_blurred_img = blur_detect_faces(haar_face_cascade, img)
        plt.imshow(convert_to_rgb(faces_blurred_img))
        plt.show()

       # face detection
        faces_detected_img = detect_faces(haar_face_cascade, img)
        plt.imshow(convert_to_rgb(faces_detected_img))
        plt.show()
