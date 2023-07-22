import argparse
import multiprocessing
from lib.people_counter import PeopleCounter, EntranceDirection
from lib.trackers import ConcreteCentroidTracker, ConcreteSortTracker
from lib.videostream import VideoStreamFromFile, VideoStreamFromDevice
from lib.updown_event import UpDownEventHandler

def parse_arguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to Caffe pre-trained model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to optional input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to optional output video file")
    ap.add_argument("-c", "--confidence", type=float, default=0.9,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=15,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())
    return args

def handle_up_down_event(data):
    print('Event: ', data)

if __name__ == '__main__':

    args = parse_arguments()

    videostream = VideoStreamFromFile(args["input"])

    # tracker = ConcreteCentroidTracker(max_disappeared=30, max_distance=50)

    # tracker = ConcreteSortTracker(max_age=40, min_hits=5)

    tracker = ConcreteSortTracker(max_age=30, min_hits=3)

    people_counter = PeopleCounter(
                                model_path=args["model"],
                                confidence=args["confidence"],
                                num_threads=multiprocessing.cpu_count(),
                                videostream=videostream,
                                skip_frames=args["skip_frames"],
                                output_file=args["output"],
                                object_tracker=tracker,
                                entrance_border=0.5,
                                entrance_direction=EntranceDirection.BOTTOM_TO_TOP,
                                up_down_handler=(UpDownEventHandler(handle_up_down_event, 5))
                                )

    print("Iniciando contagem...")
    people_counter.start_counting()
    print("Contagem finalizada...")
    print("FPS: ", people_counter.get_current_fps())

"""
TODO

- update_open_event -> increment logic

- add generic tracker (ok)
- entrance_border (ok)
- entrance_direction (TOP_TO_BOTTOM | BOTTOM_TO_TOP) (ok)
- adaptar para 0 skip_frames (ok)
- get stream of rpi camera
- tratar mesmo objeto sobe e desce 
"""