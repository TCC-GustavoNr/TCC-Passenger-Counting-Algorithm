import os
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_JSON_DIR = './runs/2023_12_03-03_53_37'
GLOBAL_METRICS = ['precision', 'recall', 'avg_fps']

def plot_metrics():
    rjson_file = open(f"{RESULTS_JSON_DIR}/results.json")
    results_json = json.load(rjson_file)
    rjson_file.close()
    
    skip_frames = []
    tracker_metrics = {} # { trackerconfig: { metricname: [] } }

    for skip_key in results_json:
        skip_num = int(skip_key.split('_')[0])
        skip_num = 0 if skip_num == 1 else skip_num
        skip_frames.append(skip_num)

        for trackerconf in results_json[skip_key]:
            if trackerconf not in tracker_metrics:
                tracker_metrics[trackerconf] = {}
                for metric in GLOBAL_METRICS:
                    tracker_metrics[trackerconf][metric] = []
            for metric in GLOBAL_METRICS:
                tracker_metrics[trackerconf][metric].append(results_json[skip_key][trackerconf]['global'][metric])
    
    for metric in GLOBAL_METRICS:
        maxy = -1
        miny = 10000
        for trackerconf in tracker_metrics:
            plt.plot(skip_frames, tracker_metrics[trackerconf][metric], label=trackerconf)
            maxy = max(maxy, max(tracker_metrics[trackerconf][metric]))
            miny = min(miny, max(tracker_metrics[trackerconf][metric]))
        if metric == 'avg_fps':
            plt.yticks(np.arange(0, maxy+10, 10))
        if metric == 'recall':
            plt.yticks(np.arange(0, 1.1, 0.1))
        if metric == 'precision':
            plt.yticks(np.arange(0.75, 1.025, 0.025))
        plt.xlabel("Skip Frames")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(f"Skip Frames x {metric.capitalize()}")
        plt.savefig(f"{RESULTS_JSON_DIR}/{metric}")
        plt.close()

if __name__ == '__main__':
    plot_metrics()
