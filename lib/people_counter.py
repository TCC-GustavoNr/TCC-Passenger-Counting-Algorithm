import cv2
import threading
import numpy as np
from enum import Enum
from typing import List
from imutils.video import FPS
from lib.debounce import debounce
from lib.videostream import AbstractVideoStream
from lib.updown_event import UpDownEvents, UpDownEventHandler
from lib.trackers import AbstractTracker, DetectedObject, TrackedObject, ObjectTracking
from tensorflow.lite.python.interpreter import Interpreter

class EntranceDirection(Enum):
    TOP_TO_BOTTOM = 1 
    BOTTOM_TO_TOP = 2

class PeopleCounter:
    def __init__(self,
                 model_path=None,
                 conf_thresh=0.6,
                 num_threads=1,
                 stop_event: threading.Event=None,
                 videostream:AbstractVideoStream=None,
                 skip_frames=5,
                 log_file=None,
                 output_file=None,
                 entrance_border=0.5,
                 entrance_direction:EntranceDirection=EntranceDirection.TOP_TO_BOTTOM,
                 object_tracker:AbstractTracker=None,
                 up_down_handler:UpDownEventHandler=None,
                ):
        
        if model_path is None or videostream is None or object_tracker is None:
            raise ValueError("Error: model_path, videostream, object_tracker attributes are invalid.")
        
        # Initializations
        self.fps = None
        self.labels = ["person"]
        self.log_file = log_file
        self.stop_event = stop_event
        self.videostream = videostream
        self.conf_thresh = conf_thresh
        self.skip_frames = skip_frames
        self.output_file = output_file
        self.object_tracker = object_tracker
        self.entrance_border = entrance_border
        self.entrance_direction = entrance_direction
        self.avg_fps = 0
        self.count_frames = 0
        self.total_up = 0
        self.total_down = 0
        self.video_width = None
        self.video_height = None
        self.video_writer = None
        self.updown_events = UpDownEvents()
        self.object_tracking = {} # ObjectTracking
        
        # Load the Tensorflow Lite model into memory
        self.mdl_interpreter = Interpreter(model_path=model_path,num_threads=num_threads)
        self.mdl_interpreter.allocate_tensors()

        # Get model details
        self.mdl_input_details = self.mdl_interpreter.get_input_details()
        self.mdl_output_details = self.mdl_interpreter.get_output_details()
        self.mdl_height = self.mdl_input_details[0]['shape'][1]
        self.mdl_width = self.mdl_input_details[0]['shape'][2]
        self.mdl_float_input = (self.mdl_input_details[0]['dtype'] == np.float32)
        self.mdl_input_mean = 127.5
        self.mdl_input_std = 127.5
        
        # Up Down event handler definition
        self.up_down_event_handler = None
        if up_down_handler: 
            @debounce(up_down_handler.debouncing_time)
            def handler():
                event_data = self.updown_events.close_open_event()
                up_down_handler.handler(event_data)
            self.up_down_event_handler = handler
        
        # Counter Summary: <tracker_info>, <skip_frames>, <num_threads>, <conf_thresh>, <entrance_border>, <entrance_direction>
        self.counter_summary = f'{object_tracker}, {skip_frames}, {num_threads}, {conf_thresh}, {entrance_border}, {entrance_direction}\n'

    def start_counting(self):
        # Open log file
        self.log_file_handler = open(self.log_file, "w", buffering=-1)
        self.log_file_handler.write(self.counter_summary)

        # Start the frames per second throughput estimator
        self.fps = FPS().start()

        while not self.stop_event.is_set():

            detected_objects: List[DetectedObject] = None

            frame = self.videostream.read()
            
            if frame is None:
                print('Frame is None!')
                break

            # Convert the frame to a blob and pass the blob through the network and obtain the detections
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image_resized = cv2.resize(image_rgb, (self.mdl_width, self.mdl_height))
            input_data = np.expand_dims(image_resized, axis=0)

            # Set video frame dimensions
            self.video_width = frame.shape[1]
            self.video_height = frame.shape[0]

            # Initialize the writer
            if self.output_file is not None and self.video_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                self.video_writer = cv2.VideoWriter(self.output_file, fourcc, 30, (self.video_width, self.video_height), True)

            # Check to see if we should run a more computationally expensive object detection method to aid our tracker
            if self.count_frames % self.skip_frames == 0:
                detected_objects = []
                
                # Object Detection
                boxes, classes, scores = self._tflite_detection(input_data)

                # Loop over all detections and draw detection box if confidence is above minimum threshold
                for i in range(len(scores)):
                    if ((scores[i] > self.conf_thresh) and (scores[i] <= 1.0)):
                        # Get bounding box - Interpreter can return coordinates that are outside of image dimensions, 
                        # need to force them to be within image using max() and min()
                        ymin = int(max(1, (boxes[i][0] * self.video_height)))
                        xmin = int(max(1, (boxes[i][1] * self.video_width)))
                        ymax = int(min(self.video_height, (boxes[i][2] * self.video_height)))
                        xmax = int(min(self.video_width, (boxes[i][3] * self.video_width)))

                        # Append the new detected object
                        detected_objects.append(DetectedObject((xmin, ymin), (xmax, ymax)))

                        # Draw object bounding box
                        if self.output_file is not None:
                            # label = f'{self.labels[int(classes[i])]}: {int(scores[i]*100)}%' # Example: 'person: 72%'
                            self._draw_bounding_box(frame, (xmin, ymin), (xmax, ymax))
                        
            # Use the tracker to associate the old object with the newly computed object
            tracked_objects = self.object_tracker.update(detected_objects, image_rgb)
            
            # Count up and down of currently tracked objects
            count_up, count_down = self._count_updown_of_tracked_objects(tracked_objects)

            # Update total counts
            self.total_up += count_up
            self.total_down += count_down

            # Check and handle up-down event
            if count_up != 0 or count_down != 0:
                print("Count update: ", count_up, count_down)
                if not self.updown_events.has_open_event():
                    self.updown_events.register_new_event()

                self.updown_events.update_open_event(count_up, count_down)
                
                if self.up_down_event_handler is not None:
                    self.up_down_event_handler()
            
            # Draw centroid of tracked objects
            if self.video_writer is not None:
                for object in tracked_objects:
                    self._draw_centroid(frame, object.centroid, f'ID:{object.object_id}')

            # Draw static/fixed contents
            if self.video_writer is not None:
                self._draw_state_info(frame)
                self._draw_entrance_border(frame)

            # Check to see if we should write the frame to disk
            if self.video_writer is not None:
                self.video_writer.write(frame)

            # Increment the total number of frames processed
            self.count_frames += 1

            # Update fps counter
            self.fps.update()
            
            # Update average fps 
            self.avg_fps = self.avg_fps + 1.0/self.count_frames * (self.get_current_fps() - self.avg_fps)

            # Update log file
            if self.log_file is not None:
                self._update_log_file(tracked_objects)

        # Stop fps timer
        self.fps.stop()

        # Release videostream
        self.videostream.release()

        # Close log file
        self.log_file_handler.close()
        
        # Print results
        print('===========================================')
        print(f'AVG FPS: {self.avg_fps}')
        print(f'Count: Up={self.total_up} Down={self.total_down}')
        print('===========================================')

    def get_current_fps(self):
        fps = None
        if self.fps is not None:
            self.fps.stop()
            fps = round(self.fps.fps())
        return fps
        
    def _tflite_detection(self, input_data):
        # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
        if self.mdl_float_input:
            input_data = (np.float32(input_data) - self.mdl_input_mean) / self.mdl_input_std

        # Perform the actual detection by running the model with the image as input
        self.mdl_interpreter.set_tensor(self.mdl_input_details[0]['index'], input_data)
        self.mdl_interpreter.invoke()

        # Retrieve detection results
        boxes = self.mdl_interpreter.get_tensor(self.mdl_output_details[1]['index'])[0] # Bounding box coordinates of detected objects
        classes = self.mdl_interpreter.get_tensor(self.mdl_output_details[3]['index'])[0]  # Class index of detected objects
        scores = self.mdl_interpreter.get_tensor(self.mdl_output_details[0]['index'])[0]  # Confidence of detected objects

        return boxes, classes, scores
    
    def _count_updown_of_tracked_objects(self, tracked_objects: List[TrackedObject]):
        entrance_border_y = round(self.entrance_border * self.video_height)
        count_up = count_down = 0

        for tracked_object in tracked_objects:
            centroid = tracked_object.centroid
            object_id = tracked_object.object_id
            tckb_obj = self.object_tracking.get(object_id, None)

            if tckb_obj is None:
                tckb_obj = ObjectTracking(object_id, centroid)
            else:
                # The difference between the y-coordinate of the *current* centroid and the mean of *previous* centroids 
                # will tell us in which direction the object is moving (negative for 'up' and positive for 'down').
                y = [c[1] for c in tckb_obj.centroids]
                tckb_obj.centroids.append(centroid)

                direction = centroid[1] - np.mean(y)
                crossed   = True if np.min(y) < entrance_border_y and np.max(y) > entrance_border_y else False
                went_up   = (crossed and 
                            (self.entrance_direction == EntranceDirection.BOTTOM_TO_TOP and direction < 0 or 
                             self.entrance_direction == EntranceDirection.TOP_TO_BOTTOM and direction > 0))
                went_down = (crossed and 
                            (self.entrance_direction == EntranceDirection.BOTTOM_TO_TOP and direction > 0 or 
                             self.entrance_direction == EntranceDirection.TOP_TO_BOTTOM and direction < 0))

                if not tckb_obj.counted:
                    if went_up:
                        count_up += 1
                        tckb_obj.counted = True
                    elif went_down:
                        count_down += 1
                        tckb_obj.counted = True

            # Store the trackable object in our dictionary
            self.object_tracking[object_id] = tckb_obj
        
        return count_up, count_down

    def _update_log_file(self, tracked_objects: List[TrackedObject]):
        try:
            logline = f'{self.count_frames} {self.get_current_fps()} {self.total_up} {self.total_down}'
            
            for obj in tracked_objects:
                logline += f', {obj.object_id} {round(obj.box_start[0])} {round(obj.box_start[1])} {round(obj.box_end[0])} {round(obj.box_end[1])}' 

            self.log_file_handler.write(logline + '\n')
        except Exception as e:
            print(f'failed to write to log file: {e}')
        

    def _draw_bounding_box(self, frame, pmin, pmax, label=None, color=(0, 255, 0)):
        xmin, ymin = pmin
        xmax, ymax = pmax

        # Draw bounding box
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        
        # Skip draw label
        if label is None:
            return
        
        # Draw label
        labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)  # Get font size

        # Make sure not to draw label too close to top of window
        label_ymin = max(ymin, labelSize[1] + 10)
        
        # Draw white box to put label text in
        cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED)
        
        # Draw label text
        cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  

    def _draw_centroid(self, frame, centroid, label, color=(0, 255, 0)):
        px, py = centroid
        cv2.putText(frame, label, (px - 10, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.circle(frame, (px, py), 4, color, -1)
        # cv2.circle(frame, (px, py), 4 + self.centroid_tracker.maxDistance, (0, 0, 255), 2)

    def _draw_entrance_border(self, frame, color=(0, 255, 0)):
        # Draw entrance border - once an object crosses this line we will determine whether they were moving 'up' or 'down'
        entrance_border_y = round(self.entrance_border * self.video_height)
        cv2.line(frame, (0, entrance_border_y), (self.video_width, entrance_border_y), color, 2)

    def _draw_state_info(self, frame, color=(0, 255, 0)):
        info = [
            ("Skip_Frames", self.skip_frames), 
            ("FPS", self.get_current_fps()), 
            ("Down", self.total_down), 
            ("Up", self.total_up)
            ]
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, self.video_height - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)