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
        
        print("TODO")

    def start_counting():
        print("TODO")

    def stop_counting():
        print("TODO")

    def get_current_fps():
        print("TODO")