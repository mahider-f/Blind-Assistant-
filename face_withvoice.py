import face_recognition
import imutils
import pickle
import time
import cv2
import os
import pytesseract
import sys
from PIL import Image
import pyttsx3
import dlib
import threading


## load the harcaascade in the cascade classifier
faceCascade = cv2.CascadeClassifier("/home/pi/Desktop/Download/haarcascade_frontalface_default.xml")# put your path of haarcascade_frontalface_default.xml
# load the known faces and embeddings saved in last file
data = pickle.loads(open('/home/pi/Desktop/encoding_yibe_exp', "rb").read())#put your path of encoding.pickle

detector = dlib.get_frontal_face_detector()

print("Streaming started")
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH,150) #resolution of the Width
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT,150) #resolution of the height
# loop over frames from the video file stream
time.sleep(0.1)
def convert_and_trim_bb(image, rect):
    # extract the starting and ending (x, y)-coordinates of the
    # bounding box
    startX = rect.left()
    startY = rect.top()
    endX = rect.right()
    endY = rect.bottom()
    # ensure the bounding box coordinates fall within the spatial
    # dimensions of the image
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])
    # compute the width and height of the bounding box
    w = endX - startX
    h = endY - startY
    # return our bounding box coordinates
    return (startX, startY, w, h)

def threaded_function():
    while True:
        # grab the frame from the threaded video stream
        ret, frame = video_capture.read()
        frame = imutils.resize(frame, width=300) # resize the size of the image frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        rects = detector(gray)
        name = "Unknown"
        if rects:
            # the facial embeddings for face in input
            boxes = face_recognition.face_locations(gray,model="hog")
            encodings = face_recognition.face_encodings(frame,boxes)
            names = []
            
            for encoding in encodings:
                # Compare encodings with encodings in data["encodings"]
                # Matches contain array with boolean values and True for the embeddings it matches closely and False for rest
                matches = face_recognition.compare_faces(data["encodings"],encoding,tolerance=0.45)
               
                if True in matches:
                    # Find positions at which we get True and store them
                    matchedIdxs = [i for (i, b) in enumerate(matches) if b]
                    counts = {}
                    # loop over the matched indexes and maintain a count for
                    # each recognized face face
                    for i in matchedIdxs:
                        # Check the names at respective indexes we stored in matchedIdxs
                        name = data["names"][i]
                        # increase count for the name we got
                        counts[name] = counts.get(name, 0) + 1
                    # set name which has highest count
                    name = max(counts, key=counts.get)
                # update the list of names
                names.append(name)
                boxe = [convert_and_trim_bb(gray, r) for r in rects]
    
                # loop for the frame
                for (x, y, w, h) in boxe:
                    # draws frame for the images
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

            print(name)
            text_speech = pyttsx3.init('espeak')
            voices = text_speech.getProperty("voices")

            text_speech.setProperty("rate", 150)
            text_speech.setProperty("volume", 0.5)

            text_speech.say(name)
            text_speech.runAndWait()

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()
    cv2.destroyAllWindows()
threaded_function()

