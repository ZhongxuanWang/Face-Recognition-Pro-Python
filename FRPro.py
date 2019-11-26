import face_recognition
import cv2
import os
from os.path import isfile, join
from random import random
import numpy as np


'''
Any changes made will ruin the app. If you still want to use it, don't do anything
'''


def get_data_face():
    # Face encoding data & Data name
    get_data_face.data_face_encodings = []
    get_data_face.name_face_encodings = []

    count1 = 0
    count2 = 0
    path_data_face = os.getcwd() + "/face_data/"

    path_data_maybe_faces = [f for f in os.listdir(
        path_data_face) if isfile(join(path_data_face, f))]

    # Get data from image folder first. < register by users themselves >
    for path_data_maybe_face in path_data_maybe_faces:
        ext = path_data_maybe_face.split(
            ".")[len(path_data_maybe_face.split(".")) - 1].lower()
        if ext == 'jpeg' or ext == 'jpg':
            count1 += 1
            # Get data name & clean path
            name_base_face_encoding = os.path.basename(
                path_data_maybe_face).split(".")[0]
            data_base_face_encoding = face_recognition.face_encodings(face_recognition.load_image_file
                                                                      (path_data_face + path_data_maybe_face))
            if len(data_base_face_encoding) == 0:
                print(name_base_face_encoding, 'face is NOT detected! *')
                continue

            print(name_base_face_encoding, 'face is detected!')

            get_data_face.data_face_encodings += [data_base_face_encoding]
            get_data_face.name_face_encodings += [name_base_face_encoding]

            count2 += 1
    print("")
    print(count1, "faces found.", count2, "faces imported.",
          str(count1 - count2), "faces were not proceeded")


def ini():
    print("\nSoftware initializing... It may takes a while.\n")
    get_data_face()


ini()


print("Press 'r' to register the Unknown face.\nPress 'q' to quit. \n\nMultiple faces are OK. But in register, first Unknown face will be proceeded")
# Initialize some variables

global match
face_locations = []
face_encodings = []
face_names = []
reg_just_now = True
reg_can_began = False
mytoken = ""


video_capture = cv2.VideoCapture(0)


while True:
    # Grab a single frame of video
    ret, frame = video_capture.read()
    frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_small_frame = small_frame[:, :, ::-1]

    # Only process every other frame of video to save time
    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(
        rgb_small_frame, face_locations)

    face_names = []

    # It's fucking strange to have a built-in method that returns a list of boolean values..........
    for position1, face_encoding in enumerate(face_encodings):
        if reg_just_now:
            break
        for position2, known_faces in enumerate(get_data_face.data_face_encodings):
            match = face_recognition.compare_faces(
                known_faces, face_encoding, tolerance=0.45)
            if True in match:
                reg_can_began = False
                face_names.append(get_data_face.name_face_encodings[position2])
                continue
            else:
                # Check if every faces have been checked
                if position2 == len(get_data_face.data_face_encodings) - 1:
                    face_names.append("Unknown")
                if reg_can_began:
                    reg_can_began = False
                    print("Face", position1,
                          "hasn't registered yet. Going to register?")
                    # cv2.destroyAllWindows()
                    register_decision = input("(y/n):").lower()
                    if register_decision != "y":
                        continue
                    while True:
                        name_unregistered = input("Input your name\n>")
                        data_face_name = os.getcwd() + "/face_data/" + name_unregistered + ".jpeg"
                        if os.path.exists(data_face_name) or len(name_unregistered) == 0:
                            print(
                                'Face name was already taken or you didn\'t input anything... Please pick up '
                                'something else')
                        else:
                            top, right, bottom, left = face_locations[position1]
                            face_img_croped = frame[top *
                                                    4:bottom * 4, left * 4:right * 4]
                            cv2.imwrite(os.getcwd() + "/face_data/" +
                                        name_unregistered + ".jpeg", face_img_croped)
                            break
                    get_data_face.data_face_encodings += [[face_encoding]]
                    get_data_face.name_face_encodings += [name_unregistered]
                    print('Congratulations! You have successfully registered.')
                    reg_just_now = True
                    break

    if reg_just_now:
        reg_just_now = False
        continue
    if len(get_data_face.name_face_encodings) == 0 and len(face_encodings) != 0 and reg_can_began:
        print("Start by giving me your name please")
        name_unregistered = input(">")
        get_data_face.name_face_encodings += []
        top, right, bottom, left = face_locations[0]
        face_img_croped = frame[top * 4:bottom * 4, left * 4:right * 4]
        cv2.imwrite(os.getcwd() + "/face_data/" +
                    name_unregistered + ".jpeg", face_img_croped)
        get_data_face.data_face_encodings += [[face_encodings[0]]]
        get_data_face.name_face_encodings += [name_unregistered]

    # Display the results
    if(len(face_names) == 0):
        face_names = ["No Data"]
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size\
        left *= 4
        top *= 4
        right *= 4
        bottom *= 4

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        size = (bottom - top) // 6

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - size),
                      (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6),
                    font, size / 30, (255, 255, 255), 1)

    # Display the resulting image
    cv2.imshow('Video', frame)

    # Hit 'q' on the keyboard to quit
    key = cv2.waitKey(5) & 0xFF
    if key == ord("q"):
        break
    if key == ord("r"):
        reg_can_began = True


# Release handle to the webcam
video_capture.release()
cv2.destroyAllWindows()