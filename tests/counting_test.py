import sys
import os
sys.path.append("..")
import json
import copy
import itertools
import numpy as np
import multiprocessing
from time import gmtime, strftime, sleep
import lib.trackers as libtrackers
from lib.videostream import VideoStreamFromFile
from lib.people_counter import PeopleCounter, EntranceDirection

MODEL_PATH = "../mobilenet_ssd/v2/detect.tflite"
CONF_THRESH = 0.9

DATASET_DIR = "../../../test_dataset"
DATASET_LABELS_PATH = f"{DATASET_DIR}/labels.txt"
DATASET_CONFIGS_PATH = "./dataset_configs.json"
TRACKER_CONFIGS_PATH = "./tracker_configs.json"
ENTRANCE_CONFIGS_PATH = "./entrance_configs.json"

TEST_SKIP_FRAMES = [1, 5, 15, 20, 25, 30]

def build_tracker(tracker_config):
    tracker = None
    
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

def build_dataset_test_dict(dataset_configs):
    labels_file = open(DATASET_LABELS_PATH)
    labels_lines = labels_file.readlines()
    dataset_labels = list(filter(lambda x: x != '\n', labels_lines))
    labels_file.close()

    dataset_test_dict = {}
    for dataset_config in dataset_configs:
        dataset_name = dataset_config['dataset_name']
        dataset_test_dict[dataset_name] = {      
            'global': {
                'avg_fps': 0,
                'std_fps': 0,
                'precision': 0, 
                'recall': 0,
                'tp': 0, 'fp': 0, 'fn': 0,
                'entering': {'precision': 0, 'recall': 0},
                'exiting': {'precision': 0, 'recall': 0},
            },
            'cumulative': {
                'fps': [],
                'entering': {'tp': 0, 'fp': 0, 'fn': 0},
                'exiting': {'tp': 0, 'fp': 0, 'fn': 0},
            },
            'videos': [],
        }

        for folder_config in dataset_config['folder_configs']:
            for video_label in dataset_labels:
                video_path, ent, ext, _ = video_label.split(' ')
                
                if folder_config['folder'] not in video_path:
                    continue
                
                int_ent = int(ent)
                int_ext = int(ext)
                dataset_test_dict[dataset_name]['videos'].append({
                    'avg_fps': 0,
                    'video_path': f'{DATASET_DIR}/{video_path}',
                    'configs': folder_config,
                    'prediction': {'entering': 0, 'exiting': 0},
                    'ground_truth': {'entering': int_ent, 'exiting': int_ext},
                })
    
    return dataset_test_dict

def get_results_from_log(logfile_path):
    sum_fps = 0
    count_frames = 0
    
    logf = open(logfile_path)
    log_lines_tmp = logf.readlines()[1:] # skip first line
    log_lines = list(filter(lambda x: x != '\n', log_lines_tmp))
    logf.close()
    
    for log_line in log_lines:
        # <frame_number> <fps> <entering_number> <exiting_number>, <object_id> <xmin> <ymin> <xmax> <ymax>, ...
        _, fps, __, ___ = log_line.split(',')[0].split(' ')
        sum_fps += int(fps)
        count_frames += 1
    
    lastline = log_lines[-1]
    _, __, ent, ext = lastline.split(',')[0].split(' ')
    
    entering = int(ent)
    exiting = int(ext)

    return entering, exiting, sum_fps/count_frames

def non_zero(value):
    return value if value != 0 else 1 

def calc_global_metrics(cumulative_metrics):
    global_metrics = {
                'avg_fps': 0, 'std_fps': 0,
                'precision': 0, 'recall': 0,
                'tp': 0, 'fp': 0, 'fn': 0,
                'entering': {'precision': 0, 'recall': 0},'exiting': {'precision': 0, 'recall': 0}
    }
    
    global_metrics['avg_fps'] = np.average(cumulative_metrics['fps'])
    global_metrics['std_fps'] = np.std(cumulative_metrics['fps'])

    global_metrics['entering']['precision'] = cumulative_metrics['entering']['tp'] / non_zero(cumulative_metrics['entering']['tp'] + cumulative_metrics['entering']['fp'])
    global_metrics['entering']['recall'] = cumulative_metrics['entering']['tp'] / non_zero(cumulative_metrics['entering']['tp'] + cumulative_metrics['entering']['fn'])
    
    global_metrics['exiting']['precision'] = cumulative_metrics['exiting']['tp'] / non_zero(cumulative_metrics['exiting']['tp'] + cumulative_metrics['exiting']['fp'])
    global_metrics['exiting']['recall'] = cumulative_metrics['exiting']['tp'] / non_zero(cumulative_metrics['exiting']['tp'] + cumulative_metrics['exiting']['fn'])
    
    global_metrics['tp'] = cumulative_metrics['entering']['tp'] + cumulative_metrics['exiting']['tp']
    global_metrics['fp'] = cumulative_metrics['entering']['fp'] + cumulative_metrics['exiting']['fp']
    global_metrics['fn'] = cumulative_metrics['entering']['fn'] + cumulative_metrics['exiting']['fn']
    
    global_metrics['precision'] = global_metrics['tp'] / non_zero(global_metrics['tp'] + global_metrics['fp'])
    global_metrics['recall'] = global_metrics['tp'] / non_zero(global_metrics['tp'] + global_metrics['fn'])
    
    return global_metrics

def write_report_file(filepath, cumulative_metrics, global_metrics):
    report_content = ''
    report_content += '========================================\n'
    report_content += f"FPS Metrics\n"
    report_content += f"FPS: {cumulative_metrics['fps']}\n"
    report_content += f"AVG FPS: {global_metrics['avg_fps']}\n"
    report_content += f"STD FPS: {global_metrics['std_fps']}\n"
    report_content += '========================================\n'
    report_content += f"Entering Metrics\n"
    report_content += f"Precision: {global_metrics['entering']['precision']}\n"
    report_content += f"Recall: {global_metrics['entering']['recall']}\n"
    report_content += f"True Positive: {cumulative_metrics['entering']['tp']}\n"
    report_content += f"False Positive: {cumulative_metrics['entering']['fp']}\n"
    report_content += f"False Negative: {cumulative_metrics['entering']['fn']}\n"
    report_content += f"Ground Truth: {cumulative_metrics['entering']['tp'] + cumulative_metrics['entering']['fn']}\n"
    report_content += '========================================\n'
    report_content += f"Exiting Metrics\n"
    report_content += f"Precision: {global_metrics['exiting']['precision']}\n"
    report_content += f"Recall: {global_metrics['exiting']['recall']}\n"
    report_content += f"True Positive: {cumulative_metrics['exiting']['tp']}\n"
    report_content += f"False Positive: {cumulative_metrics['exiting']['fp']}\n"
    report_content += f"False Negative: {cumulative_metrics['exiting']['fn']}\n"
    report_content += f"Ground Truth: {cumulative_metrics['exiting']['tp'] + cumulative_metrics['exiting']['fn']}\n"
    report_content += '========================================\n'
    report_content += f"Entering & Exiting Metrics (Total Counting)\n"
    report_content += f"Precision: {global_metrics['precision']}\n"
    report_content += f"Recall: {global_metrics['recall']}\n"
    report_content += f"True Positive: {global_metrics['tp']}\n"
    report_content += f"False Positive: {global_metrics['fp']}\n"
    report_content += f"False Negative: {global_metrics['fn']}\n"
    report_content += f"Ground Truth: {global_metrics['tp'] + global_metrics['fn']}\n"
    report_content += '========================================\n'

    report_file = open(filepath, 'w')
    report_file.write(report_content)
    report_file.close()


def run_test():
    dataset_configs_file = open(DATASET_CONFIGS_PATH)
    dataset_configs = json.load(dataset_configs_file)['configs']
    dataset_configs_file.close()

    entrance_configs_file = open(ENTRANCE_CONFIGS_PATH)
    entrance_configs = json.load(entrance_configs_file)['configs']
    entrance_configs_file.close()

    tracker_configs_file = open(TRACKER_CONFIGS_PATH)
    tracker_configs = json.load(tracker_configs_file)['configs']
    tracker_configs_file.close()

    run_folder = strftime("%Y_%m_%d-%H_%M_%S", gmtime())

    dataset_test_dict = build_dataset_test_dict(dataset_configs)

    results_dict = {}

    for skip in TEST_SKIP_FRAMES:
        results_dict[f'{skip}_skip_frames'] = {}
        for tracker_config_name in tracker_configs:
            tracker_config = tracker_configs[tracker_config_name]
            datasets_dict = copy.deepcopy(dataset_test_dict)
            tracker_dict = {
                'global': {
                    'avg_fps': 0,
                    'std_fps': 0,
                    'precision': 0, 
                    'recall': 0,
                    'tp': 0, 'fp': 0, 'fn': 0,
                    'entering': {'precision': 0, 'recall': 0},
                    'exiting': {'precision': 0, 'recall': 0},
                },
                'cumulative': {
                    'fps': [],
                    'entering': {'tp': 0, 'fp': 0, 'fn': 0},
                    'exiting': {'tp': 0, 'fp': 0, 'fn': 0},
                }, 
            }

            for dataset_config in dataset_configs:
                dataset_name = dataset_config['dataset_name']
                test_path = f'./runs/{run_folder}/{skip}_skip_frames/{tracker_config_name}/{dataset_name}/logs'
                os.makedirs(test_path, exist_ok=True)

                for dataset_video in datasets_dict[dataset_name]['videos']:
                    entrance_config = entrance_configs[dataset_video['configs']['entrance_config']]
                    videoname = dataset_video['video_path'].split('/')[-1].split('.')[0]
                    logfile = f'{test_path}/{videoname}.txt'
                    tracker = build_tracker(tracker_config)
                    videostream = VideoStreamFromFile(dataset_video['video_path'])
                    
                    entrance_direction = None
                    if EntranceDirection.TOP_TO_BOTTOM.value == entrance_config['entrance_direction']:
                        entrance_direction = EntranceDirection.TOP_TO_BOTTOM
                    else:
                        entrance_direction = EntranceDirection.BOTTOM_TO_TOP

                    print(f"\tTEST - Skip Frames: {skip} | Tracker Config: {tracker_config_name} | Dataset: {dataset_config['dataset_name']} | Video: {videoname}")

                    people_counter = PeopleCounter(
                        model_path=MODEL_PATH,
                        conf_thresh=CONF_THRESH,
                        num_threads=multiprocessing.cpu_count(),
                        videostream=videostream,
                        skip_frames=skip,
                        log_file=logfile,
                        output_file=None,
                        object_tracker=tracker,
                        entrance_border=entrance_config['entrance_border'],
                        entrance_direction=entrance_direction,
                        up_down_handler=None
                    ) 

                    # run counter
                    people_counter.start_counting()
                    
                    # get and save results
                    ent, ext, avg_fps = get_results_from_log(logfile)
                    dataset_video['avg_fps'] = avg_fps
                    dataset_video['prediction']['entering'] = ent
                    dataset_video['prediction']['exiting'] = ext

                    # cumulative metrics - fps
                    datasets_dict[dataset_name]['cumulative']['fps'].append(avg_fps)

                    # cumulative metrics - entering 
                    err = dataset_video['prediction']['entering'] - dataset_video['ground_truth']['entering']
                    if err == 0:
                        datasets_dict[dataset_name]['cumulative']['entering']['tp'] += dataset_video['prediction']['entering']
                    elif err < 0:
                        datasets_dict[dataset_name]['cumulative']['entering']['tp'] += dataset_video['prediction']['entering']
                        datasets_dict[dataset_name]['cumulative']['entering']['fn'] += abs(err)
                    elif err > 0:
                        datasets_dict[dataset_name]['cumulative']['entering']['tp'] += dataset_video['ground_truth']['entering']
                        datasets_dict[dataset_name]['cumulative']['entering']['fp'] += abs(err)

                    # cumulative metrics - exiting 
                    err = dataset_video['prediction']['exiting'] - dataset_video['ground_truth']['exiting']
                    if err == 0:
                        datasets_dict[dataset_name]['cumulative']['exiting']['tp'] += dataset_video['prediction']['exiting']
                    elif err < 0:
                        datasets_dict[dataset_name]['cumulative']['exiting']['tp'] += dataset_video['prediction']['exiting']
                        datasets_dict[dataset_name]['cumulative']['exiting']['fn'] += abs(err)
                    elif err > 0:
                        datasets_dict[dataset_name]['cumulative']['exiting']['tp'] += dataset_video['ground_truth']['exiting']
                        datasets_dict[dataset_name]['cumulative']['exiting']['fp'] += abs(err)

                # sleep
                sleep(1)

                # dataset global metrics
                datasets_dict[dataset_name]['global'] = calc_global_metrics(datasets_dict[dataset_name]['cumulative'])

                # tracker cumulative metrics
                tracker_dict['cumulative']['fps'].extend(datasets_dict[dataset_name]['cumulative']['fps'])
                tracker_dict['cumulative']['entering']['tp'] += datasets_dict[dataset_name]['cumulative']['entering']['tp'] 
                tracker_dict['cumulative']['entering']['fp'] += datasets_dict[dataset_name]['cumulative']['entering']['fp']
                tracker_dict['cumulative']['entering']['fn'] += datasets_dict[dataset_name]['cumulative']['entering']['fn']
                tracker_dict['cumulative']['exiting']['tp'] += datasets_dict[dataset_name]['cumulative']['exiting']['tp']
                tracker_dict['cumulative']['exiting']['fp'] += datasets_dict[dataset_name]['cumulative']['exiting']['fp']
                tracker_dict['cumulative']['exiting']['fn'] += datasets_dict[dataset_name]['cumulative']['exiting']['fn']

                # write dataset metrics 
                write_report_file(f'./runs/{run_folder}/{skip}_skip_frames/{tracker_config_name}/{dataset_name}/report.txt', 
                                  datasets_dict[dataset_name]['cumulative'],
                                  datasets_dict[dataset_name]['global'],
                                )

            # tracker global metrics
            tracker_dict['global'] = calc_global_metrics(tracker_dict['cumulative'])

            # save in results_dict
            tracker_dict['datasets'] = datasets_dict
            results_dict[f'{skip}_skip_frames'][tracker_config_name] = tracker_dict

            # write tracker metrics 
            write_report_file(f'./runs/{run_folder}/{skip}_skip_frames/{tracker_config_name}/report.txt',
                              tracker_dict['cumulative'],
                              tracker_dict['global'],
                            )
            
            # write in progress json results
            results_file = open(f'./runs/{run_folder}/results.json', 'w')
            results_file.write(json.dumps(results_dict, indent=4))
            results_file.close()

    # write json results
    results_file = open(f'./runs/{run_folder}/results.json', 'w')
    results_file.write(json.dumps(results_dict, indent=4))
    results_file.close()

if __name__ == '__main__':
    run_test()

