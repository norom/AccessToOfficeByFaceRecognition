import cv2
import time
import serial
import face_recognition
import numpy as np
from enum import Enum
import datetime

import os
# from os import path
from os import listdir
from os.path import isfile, join

def printLog(str):
    print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ": " + str)

class StateMachine(Enum):
    WAITING_ANYONE = "Waiting for anyone"
    RECOGNIZING = "Recognizing"
    ACCESS_DENIED = "Access denied"
    ACCESS_ALLOWED = "Please enter"


# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Get a reference to webcam #0 (the default one)
video_capture = cv2.VideoCapture(0)
video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1);
# Load a sample pictures in folder db and learn how to recognize it.
mypath = "db"
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
known_face_encodings = []
known_face_names = []
for fileName in onlyfiles:
    fullFileName = mypath + '/' + fileName
    printLog(fullFileName)
    image = face_recognition.load_image_file(fullFileName)
    face_encoding = face_recognition.face_encodings(image)[0]
    known_face_encodings.append(face_encoding)
    known_face_names.append(os.path.splitext(fileName)[0])

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
process_this_frame = True

current_state = StateMachine.WAITING_ANYONE

iteration_delay = float(0.1)
relay_delay = float(0.3)
open_delay = float(5.0)


while True:
    time.sleep(iteration_delay)
    # Grab a single frame of video
    ret, frame = video_capture.read()
    # small_frame = cv2.resize(frame, (0, 0), fx=0.15, fy=0.15)
    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    if process_this_frame:
        # Find all the faces and face encodings in the current frame of video
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        if len(face_encodings) == 0:
            current_state = StateMachine.WAITING_ANYONE
        else:
            current_state = StateMachine.RECOGNIZING
            printLog("Face detected")

        face_names = []
        for face_encoding in face_encodings:
            # See if the face is a match for the known face(s)
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
            name = "Unknown"

            # # If a match was found in known_face_encodings, just use the first one.
            # if True in matches:
            #     first_match_index = matches.index(True)
            #     name = known_face_names[first_match_index]

            # Or instead, use the known face with the smallest distance to the new face
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_face_names[best_match_index]
                face_names.append(name)

    # process_this_frame = not process_this_frame

    if current_state != StateMachine.ACCESS_ALLOWED:
        if len(face_names) > 0:
            current_state = StateMachine.ACCESS_ALLOWED
            printLog(face_names[0])

    # Display the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    cv2.putText(frame, current_state.name, (30, 30), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if current_state == StateMachine.ACCESS_ALLOWED:
        printLog("Open")
        ser = serial.Serial(port='/dev/ttyUSB0', baudrate=9600, bytesize=serial.EIGHTBITS, parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE, timeout=None, xonxoff=False, rtscts=False, write_timeout=None,
                            dsrdtr=False, inter_byte_timeout=None)

        ser.write(b'\x01\x05\x00\x00\xff\x00\x8c\x3a')  # close relay
        time.sleep(relay_delay)
        ser.write(b'\x01\x05\x00\x00\x00\x00\xcd\xca')  # open relay

        ser.close()  # close port
        #video_capture.release()
        time.sleep(open_delay)
        ret, frame = video_capture.read()
        #video_capture = cv2.VideoCapture(0)

    current_state = StateMachine.WAITING_ANYONE

# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()
