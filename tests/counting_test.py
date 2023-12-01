import sys
sys.path.append("..")
import os
import json
from time import gmtime, strftime
import lib.trackers as trackers
from lib.videostream import VideoStreamFromFile

MODEL_PATH = "../mobilenet_ssd/v2/detect.tflite"
CONF_THRESH = 0.9

DATASET_LABELS_PATH = "../../../test_dataset/labels.txt"
ENTRANCE_CONFIGS_PATH = "./entrance_configs.json"
TRACKER_CONFIGS_PATH = "./tracker_configs.json"

TEST_SKIP_FRAMES = [1, 5, 15, 20, 25, 30]


def run_test():
    entrance_configs_file = open(ENTRANCE_CONFIGS_PATH)
    entrance_configs = json.load(entrance_configs_file)['configs']
    entrance_configs_file.close()

    tracker_configs_file = open(TRACKER_CONFIGS_PATH)
    tracker_configs = json.load(tracker_configs_file)['configs']
    tracker_configs_file.close()

    datetime = strftime("%Y_%m_%d-%H_%M_%S", gmtime())
    
    # os.mkdir(f'./runs/{datetime}')
    # x_skip_frames/datasetname/trakername

    print(entrance_configs, tracker_configs)


if __name__ == '__main__':
    run_test()