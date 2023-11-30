import argparse
import multiprocessing
from lib.updown_event import UpDownEventHandler
from lib.people_counter import PeopleCounter, EntranceDirection
from lib.videostream import VideoStreamFromFile, VideoStreamFromDevice
from lib.trackers import StandardCentroidTracker, CorrelationCentroidTracker, StandardSortTracker, CorrelationSortTracker


def parse_arguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to tensorflow model")
    ap.add_argument("-i", "--input", type=str,
                    help="path to input video file")
    ap.add_argument("-o", "--output", type=str,
                    help="path to output video file (optional)")
    ap.add_argument("-c", "--confidence", type=float, default=0.9,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=15,
                    help="number of skip frames between detections")
    args = vars(ap.parse_args())
    return args


def handle_up_down_event(data):
    print('Event: ', data)


if __name__ == '__main__':

    args = parse_arguments()

    videostream = VideoStreamFromFile(args["input"])

    # tracker = CorrelationCentroidTracker(max_disappeared=30, max_distance=50)

    # tracker = CorrelationSortTracker(max_age=40, min_hits=5)

    # tracker = CorrelationSortTracker(max_age=30, min_hits=3)

    tracker = StandardSortTracker(max_age=30, min_hits=1, iou_threshold=0.2)

    # tracker = StandardCentroidTracker(max_disappeared=30, max_distance=50)

    people_counter = PeopleCounter(
        model_path=args["model"],
        conf_thresh=args["confidence"],
        num_threads=multiprocessing.cpu_count(),
        videostream=videostream,
        skip_frames=args["skip_frames"],
        log_file='./logfile.txt',
        output_file=args["output"],
        object_tracker=tracker,
        entrance_border=0.50,
        entrance_direction=EntranceDirection.BOTTOM_TO_TOP,
        up_down_handler=(UpDownEventHandler(handle_up_down_event, 5))
    )

    print(tracker)
    print("Iniciando contagem...")
    # people_counter.start_counting()
    print("Contagem finalizada.")

