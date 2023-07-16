from abc import ABC, abstractmethod
import dlib
import numpy as np
from cv2 import Mat
from typing import List
from lib.centroid_tracker import CentroidTracker
from lib.sort_tracker import Sort as SortTracker

class DetectedObject:
    def __init__(self, box_start: tuple, box_end: tuple) -> None:
        self.box_start = box_start
        self.box_end = box_end

class TrackedObject:
    def __init__(self, object_id: int, box_start: tuple, box_end: tuple) -> None:
        self.object_id = object_id
        self.box_start = box_start
        self.box_end = box_end
        self.centroid = (int((box_start[0] + box_end[0]) / 2.0), int((box_start[1] + box_end[1]) / 2.0))

class ObjectTracking:
	def __init__(self, object_id, centroid):
		self.object_id = object_id
		self.centroids = [centroid] # to store object tracking centroids (path of object)
		self.counted = False # indicate if the object has already been counted or not

class DlibCorrelationTracker:
    # The correlation_tracker objects instead of performing a redetection, thus achieving a higher frame processing rate
    def __init__(self) -> None:
        self.correlation_trackers = []

    def start_correlation_tracker(self, detections: List[DetectedObject], frame_rgb: Mat) -> None:
        self.correlation_trackers = []
        for detection in detections:
            xmin, ymin = detection.box_start
            xmax, ymax = detection.box_end
            # Construct a dlib rectangle object from the bounding box coordinates and then start the dlib correlation tracker
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(xmin, ymin, xmax, ymax)
            tracker.start_track(frame_rgb, rect)
            # Add the tracker to our list of trackers so we can utilize it during skip frames
            self.correlation_trackers.append(tracker)
    
    def update_correlation_tracker(self, frame_rgb: Mat) -> List[DetectedObject]:
        updated_positions:List[DetectedObject] = []
        for tracker in self.correlation_trackers:
            # Update the tracker and grab the updated position
            confidence = tracker.update(frame_rgb)
            pos = tracker.get_position()
            # Add the bounding box coordinates to the rectangles list
            updated_positions.append(DetectedObject((int(pos.left()), int(pos.top())), (int(pos.right()), int(pos.bottom()))))
        return updated_positions

class AbstractTracker(ABC):
    @abstractmethod
    def update(self, detections: List[DetectedObject], frame_rgb: Mat) -> List[TrackedObject]: 
        pass

class ConcreteCentroidTracker(AbstractTracker):
    def __init__(self, max_disappeared=50, max_distance=50) -> None:
        super().__init__()
        self.tracker = CentroidTracker(maxDisappeared=max_disappeared, maxDistance=max_distance)
        self.correlation_tracker = DlibCorrelationTracker()

    def update(self, detections: List[DetectedObject], frame_rgb: Mat) -> List[TrackedObject]: 
        detected_objs = []
 
        if detections:
            detected_objs = detections
            self.correlation_tracker.start_correlation_tracker(detections, frame_rgb)
        else:
            detected_objs = self.correlation_tracker.update_correlation_tracker(frame_rgb)
        
        # tracker_inputs: [(x1, y1, x2, y2), ... ]
        tracker_inputs = list(map(lambda d : (d.box_start[0], d.box_start[1], d.box_end[0], d.box_end[1]), detected_objs))
        
        # tracker_outputs: { key: (centroid(x, y), box((x1, y1),(x2, y2))), ... }
        tracker_outputs = self.tracker.update(tracker_inputs)

        tracked_objs = list(map(lambda o : TrackedObject(int(o[0]), o[1][1][0], o[1][1][1]), tracker_outputs.items()))

        return tracked_objs

class ConcreteSortTracker(AbstractTracker):
    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3) -> None:
        super().__init__()
        self.tracker = SortTracker(max_age=max_age, min_hits=min_hits, iou_threshold=iou_threshold)
        self.correlation_tracker = DlibCorrelationTracker()
    
    def update(self, detections: List[DetectedObject], frame_rgb: Mat) -> List[TrackedObject]: 
        detected_objs = []
 
        if detections:
            detected_objs = detections
            self.correlation_tracker.start_correlation_tracker(detections, frame_rgb)
        else:
            detected_objs = self.correlation_tracker.update_correlation_tracker(frame_rgb)
        
        # tracker_inputs: [[x1, y1, x2, y2, score], ... ]
        tracker_inputs = list(map(lambda d : [d.box_start[0], d.box_start[1], d.box_end[0], d.box_end[1], None], detected_objs))
        
        # tracker_outputs: [[x1, y1, x2, y2, id, score], ... ]
        tracker_outputs = self.tracker.update(np.array(tracker_inputs))

        tracked_objs = list(map(lambda o : TrackedObject(int(o[4])-1, (o[0], o[1]), (o[2], o[3])), tracker_outputs))

        return tracked_objs
