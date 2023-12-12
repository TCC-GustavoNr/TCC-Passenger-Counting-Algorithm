import os
import json
import numpy as np
import matplotlib.pyplot as plt

RESULTS_JSON_DIR = './runs/2023_12_03-03_53_37'
MAIN_METRICS = ['precision', 'recall', 'avg_fps']

def get_metrics():
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
                for metric in MAIN_METRICS:
                    tracker_metrics[trackerconf][metric] = []
            for metric in MAIN_METRICS:
                tracker_metrics[trackerconf][metric].append(results_json[skip_key][trackerconf]['global'][metric])

    max_fps = 1
    for trackerconf in tracker_metrics:
        max_fps = max(max_fps, max(tracker_metrics[trackerconf]['avg_fps']))

    for trackerconf in tracker_metrics:
        tracker_metrics[trackerconf]['arithmetic_avg'] = []
        tracker_metrics[trackerconf]['harmonic_avg'] = []
        for i in range(len(skip_frames)):
            asum = 0
            hsum = 0
            for metric in MAIN_METRICS:
                if metric == 'avg_fps':
                    asum += tracker_metrics[trackerconf][metric][i] / max_fps
                    hsum += 1.0 / (tracker_metrics[trackerconf][metric][i] / max_fps)
                else:
                    asum += tracker_metrics[trackerconf][metric][i]
                    hsum += 1.0 / tracker_metrics[trackerconf][metric][i]
            tracker_metrics[trackerconf]['arithmetic_avg'].append(asum / len(MAIN_METRICS))
            tracker_metrics[trackerconf]['harmonic_avg'].append(len(MAIN_METRICS) / hsum)
        
    return skip_frames, tracker_metrics

def plot_metrics():
    skip_frames, tracker_metrics = get_metrics()

    for metric in MAIN_METRICS:
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
    
    GLOBAL_METRICS = ['arithmetic_avg', 'harmonic_avg']

    for metric in GLOBAL_METRICS:
        for trackerconf in tracker_metrics:
            plt.plot(skip_frames, tracker_metrics[trackerconf][metric], label=trackerconf)

        plt.xlabel("Skip Frames")
        plt.ylabel(metric.capitalize())
        plt.legend()
        plt.title(f"Skip Frames x {metric.capitalize()}")
        plt.savefig(f"{RESULTS_JSON_DIR}/{metric}")
        plt.close()

if __name__ == '__main__':
    plot_metrics()
