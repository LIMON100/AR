# -*- coding: utf-8 -*-
"""
Created on Mon May  3 11:40:08 2021

@author: limon
"""

import pika
import os
import cv2
import math
import random
import numpy as np
import datetime as dt
import tensorflow as tf
from moviepy.editor import *
from collections import deque
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import plot_model

from keras.models import load_model
import os
import numpy as np
from keras.preprocessing import image
import cv2
import matplotlib.pyplot as plt


import time

#from imutils.video import FileVideoStream
#from imutils.video import FPS



def predict_on_live_video(video_file_path, output_file_path, window_size):
    
    model = load_model('cnn_model.h5')


    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    
    channel.queue_declare(queue='hello')
    
    
    image_height, image_width = 128, 128
    
    classes_list = [
        "normal driving",
        "texting - right",
        "talking on the phone - right",
        "texting - left",
        "talking on the phone - left",
        "operating the radio",
        "drinking",
        "reaching behind",
        "hair and makeup",
        "talking to passenger"
    ]

    

    # Initialize a Deque Object with a fixed size which will be used to implement moving/rolling average functionality.
    predicted_labels_probabilities_deque = deque(maxlen = window_size)

    video_reader = cv2.VideoCapture(video_file_path)
    #video_reader = FileVideoStream(video_file_path).start()
    #time.sleep(0.01)

    original_video_width = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_video_height = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    #original_video_width = int(cv2.CAP_PROP_FRAME_WIDTH)
    #original_video_height = int(cv2.CAP_PROP_FRAME_HEIGHT)

    video_writer = cv2.VideoWriter(output_file_path, cv2.VideoWriter_fourcc('M', 'P', '4', 'V'), 24, (original_video_width, original_video_height))

    while True: 

        status, frame = video_reader.read() 

        if not status:
            break

        resized_frame = cv2.resize(frame, (image_height, image_width))
 
        normalized_frame = resized_frame / 255

        predicted_labels_probabilities = model.predict(np.expand_dims(normalized_frame, axis = 0))[0]

        # Appending predicted label probabilities to the deque object
        predicted_labels_probabilities_deque.append(predicted_labels_probabilities)

        if len(predicted_labels_probabilities_deque) == window_size:

            # Converting Numpy array
            predicted_labels_probabilities_np = np.array(predicted_labels_probabilities_deque)

            predicted_labels_probabilities_averaged = predicted_labels_probabilities_np.mean(axis = 0)
            
            # Return the Maximum value
            predicted_label = np.argmax(predicted_labels_probabilities_averaged)

            predicted_class_name = classes_list[predicted_label]
            
            if predicted_class_name == "texting - right" or predicted_class_name == "texting - left" or predicted_class_name == "talking on the phone - left" or predicted_class_name == "operating the radio" or predicted_class_name == "drinking" or predicted_class_name == "talking on the phone - right":
                
                channel.basic_publish(exchange='', routing_key='hello', body='Hello World!')
                print("Sent Message!'")
                
            cv2.putText(frame, predicted_class_name, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        video_writer.write(frame)

        cv2.imshow('Predicted Frames', frame)

        key_pressed = cv2.waitKey(10)

        if key_pressed == ord('q'):
            break

    cv2.destroyAllWindows()

    video_reader.release()
    #video_reader.stop()
    video_writer.release()
    connection.close()
    


output_directory = 'Youtube_Videos'
os.makedirs(output_directory, exist_ok = True)
video_title = "mptog omg"
input_video_file_path = 'Eating_high_res.mp4'
#input_video_file_path = 0


# Setting sthe Window Size which will be used by the Rolling Average Proces
window_size = 1

output_video_file_path = f'{output_directory}/{video_title} -Output-WSize {window_size}.mp4'

predict_on_live_video(input_video_file_path, output_video_file_path, window_size)

VideoFileClip(output_video_file_path).ipython_display(width = 700)