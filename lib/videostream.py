from abc import ABC, abstractmethod
from typing import Union
import threading
import queue
import cv2
import time


class AbstractVideoStream(ABC):
    @abstractmethod
    def read(self) -> Union[cv2.UMat, None]:
        pass

    @abstractmethod
    def release(self) -> None:
        pass


class VideoStreamFromFile(AbstractVideoStream):
    """
    This class allows you to stream video from a video file.

    @param filepath
    """

    def __init__(self, filepath) -> None:
        self.cap = cv2.VideoCapture(filepath)

    def read(self) -> Union[cv2.UMat, None]:
        ret, frame = self.cap.read()
        return frame

    def release(self) -> None:
        return self.cap.release()


class VideoStreamFromDevice(AbstractVideoStream):
    """
    This class allows you to stream video from a connected device (camera).

    @param device: device index or device path (/dev/videoX)

    * Creates a thread to handle reading and storing frames in a queue.
    * Read the frames as soon as they are available this approach removes OpenCV's internal buffer and reduces the frame lag.
    """

    def __init__(self, device) -> None:
        self.cap = cv2.VideoCapture(device)
        self.q = queue.Queue()
        self.thr = threading.Thread(target=self._reader)
        self.thr.daemon = True
        self.thr.start()

    def _reader(self) -> None:
        while True:
            ret, frame = self.cap.read()  # read the frames
            if not ret:
                # close the thread - device is not working
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass
            self.q.put(frame)  # store them in a queue (instead of the buffer)

    def read(self) -> Union[cv2.UMat, None]:
        try:
            return self.q.get(block=False)
        except queue.Empty:
            return None

    def release(self) -> None:
        return self.cap.release()
