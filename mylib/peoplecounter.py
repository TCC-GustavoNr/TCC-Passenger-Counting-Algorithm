from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import time
import csv
import numpy as np
import time
import dlib
import cv2
import datetime
from itertools import zip_longest
from tensorflow.lite.python.interpreter import Interpreter

class PeopleCounter:
    def __init__(self, 
                 model_path=None, 
                 confidence=0.6, 
                 num_threads=2, 
                 videostream=None, 
                 skip_frames=5, 
                 output_path=None,
                 entrance_border_h=0.5,
                 ct_max_disappeared=60,
                 ct_max_distance=100,
                 up_down_handler=None,
                 ):
        
        if not model_path or not videostream:
            raise ValueError("Error: model_path, videostream attributes are invalid.")
        
        # Initializations
        self.curr_fps = 0
        self.mdl_labels = ["person"]
        self.mdl_confidence = confidence
        self.skip_frames = skip_frames
        self.output_path = output_path
        self.entrance_border_h = entrance_border_h

        # Load the Tensorflow Lite model into memory
        self.mdl_interpreter = Interpreter(model_path=model_path,num_threads=num_threads)
        self.mdl_interpreter.allocate_tensors()

        # Get model details
        self.mdl_input_details = self.interpreter.get_input_details()
        self.mdl_output_details = self.interpreter.get_output_details()
        self.mdl_height = self.input_details[0]['shape'][1]
        self.mdl_width = self.input_details[0]['shape'][2]
        self.mdl_float_input = (self.input_details[0]['dtype'] == np.float32)
        self.mdl_input_mean = 127.5
        self.mdl_input_std = 127.5
        
        print("TODO")

    def start_counting():
        print("TODO")

    def stop_counting():
        print("TODO")

    def get_current_fps():
        print("TODO")