import os
import json
import argparse
import threading
import multiprocessing
from time import gmtime, strftime, sleep
import lib.trackers as libtrackers
from lib.updown_event import UpDownEventHandler
from lib.people_counter import PeopleCounter, EntranceDirection
from lib.videostream import VideoStreamFromFile, VideoStreamFromDevice

class Repeat(threading.Timer):
    def run(self):
        while not self.finished.wait(self.interval):
            self.function(*self.args, **self.kwargs)

def check_stop_required(*args):
    stop_event = args[0]
    STOP_FILE_PATH = './stop_file.txt'
    print("Checking if stop is required...")
    if os.path.isfile(STOP_FILE_PATH):
        stop_event.set()
        print("Set Stop Event: Stop file setted.")

def parse_arguments():
    default_logfile = 'logs/' + strftime("%Y_%m_%d-%H_%M_%S", gmtime()) + '.txt'
    default_entrance_config = "configs/entrance_config.json"
    default_tracker_config = "configs/tracker_config.json"

    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-m", "--model", required=True,
                    help="path to tensorflow model")
    ap.add_argument("-i", "--input", type=str, required=True,
                    help="path to input video file or device number")
    ap.add_argument("-o", "--output", type=str,
                    help="path to output video file")
    ap.add_argument("-c", "--conf-thresh", type=float, default=0.9,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=10,
                    help="number of skip frames between detections")
    ap.add_argument("-l", "--log-file", type=str, default=default_logfile,
                    help="path to log file")
    ap.add_argument("-e", "--entrance-config", type=str, default=default_entrance_config,
                    help="path to json file with entrance configuration")
    ap.add_argument("-t", "--tracker-config", type=str, default=default_tracker_config,
                    help="path to json file with tracker configuration")
    args = vars(ap.parse_args())

    return args

def build_tracker(tracker_config_path):
    tracker = None

    tracker_configs_file = open(tracker_config_path)
    tracker_config = json.load(tracker_configs_file)['config']
    tracker_configs_file.close()
    
    if tracker_config['name'] == 'StandardCentroidTracker':
        tracker = libtrackers.StandardCentroidTracker(
            max_disappeared=tracker_config['params']['max_disappeared'], 
            max_distance=tracker_config['params']['max_distance']
        )
    elif tracker_config['name'] == 'CorrelationCentroidTracker':
        tracker = libtrackers.CorrelationCentroidTracker(
            max_disappeared=tracker_config['params']['max_disappeared'], 
            max_distance=tracker_config['params']['max_distance']
        )
    elif tracker_config['name'] == 'StandardSortTracker':
        tracker = libtrackers.StandardSortTracker(
            max_age=tracker_config['params']['max_age'],
            min_hits=tracker_config['params']['min_hits'],
            iou_threshold=tracker_config['params']['iou_threshold'],
        )
    elif tracker_config['name'] == 'CorrelationSortTracker':
        tracker = libtrackers.CorrelationSortTracker(
            max_age=tracker_config['params']['max_age'],
            min_hits=tracker_config['params']['min_hits'],
            iou_threshold=tracker_config['params']['iou_threshold'],
        )
    
    return tracker

def load_entrance_config(entrance_config_path):
    entrance_configs_file = open(entrance_config_path)
    entrance_config = json.load(entrance_configs_file)['config']
    entrance_configs_file.close()
    
    entrance_direction = None
    if EntranceDirection.TOP_TO_BOTTOM.value == entrance_config['entrance_direction']:
        entrance_direction = EntranceDirection.TOP_TO_BOTTOM
    else:
        entrance_direction = EntranceDirection.BOTTOM_TO_TOP
    
    entrance_config['entrance_direction'] = entrance_direction

    return entrance_config

def handle_up_down_event(data):
    print('Event: ', data)


if __name__ == '__main__':
    
    args = parse_arguments()

    videostream = None
    if args["input"].isdigit():
        videostream = VideoStreamFromDevice(f"/dev/video{args['input']}")
    else:
        videostream = VideoStreamFromFile(args["input"])

    tracker = build_tracker(args["tracker_config"])

    entrance_config = load_entrance_config(args["entrance_config"])

    stop_event = threading.Event()

    check_stop_timer = Repeat(5, check_stop_required, [stop_event])
    check_stop_timer.start()

    people_counter = PeopleCounter(
        model_path=args["model"],
        conf_thresh=args["conf_thresh"],
        num_threads=multiprocessing.cpu_count(),
        stop_event=stop_event,
        videostream=videostream,
        skip_frames=args["skip_frames"],
        log_file=args["log_file"],
        output_file=args["output"],
        object_tracker=tracker,
        entrance_border=entrance_config["entrance_border"],
        entrance_direction=entrance_config["entrance_direction"],
        up_down_handler=(UpDownEventHandler(handle_up_down_event, 5))
    )

    print("Tracker:", tracker)

    print("Iniciando contagem...")
    people_counter.start_counting()
    print("Contagem finalizada.")
    check_stop_timer.cancel()
    print("Exiting...")
