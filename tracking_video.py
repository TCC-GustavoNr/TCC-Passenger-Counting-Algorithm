import os
import cv2

"""
Log File:
<tracker_info>, <skip_frames>, <num_threads>, <conf_thresh>, <entrance_border>, <entrance_direction>
<frame_number> <fps> <entering_number> <exiting_number>, <object_id> <xmin> <ymin> <xmax> <ymax>, ...
...
<frame_number> <fps> <entering_number> <exiting_number>, <object_id> <xmin> <ymin> <xmax> <ymax>, ...
"""

INFO_TEXT_CONFIGS = {
    'pcds': {
        'top': {
            'thickness': 1,
            'font_scale': 0.6,
            'space_between': 20,
            'pad': (5, 15),
            'color': (0, 200, 0),
        },
        'bottom': {
            'thickness': 1,
            'font_scale': 0.6,
            'space_between': 20,
            'pad': (5, 10),
            'color': (0, 200, 0),
        },   
    },
    'peruibe': {
        'top': {
            'thickness': 2,
            'font_scale': 0.8,
            'space_between': 25,
            'pad': (5, 20),
            'color': (0, 200, 0),
        },
        'bottom': {
            'thickness': 2,
            'font_scale': 0.8,
            'space_between': 25,
            'pad': (5, 10),
            'color': (0, 200, 0),
        },
    },
    'mvia': {
        'top': {
            'thickness': 2,
            'font_scale': 0.8,
            'space_between': 25,
            'pad': (5, 20),
            'color': (230, 0, 0),
        },
        'bottom': {
            'thickness': 2,
            'font_scale': 0.8,
            'space_between': 25,
            'pad': (5, 10),
            'color': (230, 0, 0),
        },
    },
}

FPS_CONFIGS = {
    'pcds': 25,
    'peruibe': 30,
    'mvia': 30,
}

class TrackingVideoGenerator:    

    @classmethod
    def run(self, input_video, output_video, logfile_path, rec_fps, text_config):
        print('Generating traking video...')

        logfile = open(logfile_path, 'r')
        
        # First line
        exec_config = logfile.readline()
        str_tracker_info, str_skip_frames, _, __, str_entrance_border, ___ = exec_config.split(',')
        str_tracker_info = str_tracker_info.strip()
        skip_frames = int(str_skip_frames.strip())
        entrance_border = float(str_entrance_border.strip())

        id_count = 1 
        id_mapping = {} # { logid: out_id }

        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        videowriter = None

        videostream = cv2.VideoCapture(input_video)

        for line in logfile.readlines():
            ret, frame = videostream.read()
            
            if frame is None:
                break

            if videowriter is None:
                videowriter = cv2.VideoWriter(output_video, fourcc, rec_fps, (frame.shape[1], frame.shape[0]))

            str_frame_info, *str_objects = line.strip().split(',')
            
            _, curr_fps, entering, exiting = str_frame_info.strip().split(' ')

            bottom_info_list = [
                ("FPS", curr_fps), 
                ("Saidas", exiting), 
                ("Entradas", entering)
            ]
        
            top_info_list = [
                ("Tracker", str_tracker_info.split(' ')[0]),
                ("Skip Frames", skip_frames),
            ]
            
            for str_obj in str_objects:
                log_id, *sbbox = str_obj.strip().split(' ')
                
                if log_id not in id_mapping:
                    id_mapping[log_id] = id_count
                    id_count += 1
                
                bbox = list(map(int, sbbox))
                centroid = [bbox[0] + round(abs(bbox[0] - bbox[2])/2.0), bbox[1] + round(abs(bbox[1] - bbox[3])/2.0)]

                self._draw_centroid(self, frame, centroid, f"ID:{id_mapping[log_id]}")

            self._draw_entrance_border(self, frame, entrance_border)
            self._draw_top_info_list(self, frame, top_info_list, text_config['top'])
            self._draw_bottom_info_list(self, frame, bottom_info_list, text_config['bottom'])
            videowriter.write(frame)
        
        videostream.release()
        videowriter.release()
        logfile.close()

    def _draw_bounding_box(self, frame, bbox, label=None, color=(0, 255, 0)):
            xmin, ymin = bbox[:2]
            xmax, ymax = bbox[2:]

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

    def _draw_entrance_border(self, frame, entrance_border, color=(0, 255, 0)):
            video_width = frame.shape[1]
            video_height = frame.shape[0]

            # Draw entrance border - once an object crosses this line we will determine whether they were moving 'up' or 'down'
            entrance_border_y = round(entrance_border * video_height)
            cv2.line(frame, (0, entrance_border_y), (video_width, entrance_border_y), color, 2)

    def _draw_top_info_list(self, frame, info_list, text_config):
        video_width = frame.shape[1]
        video_height = frame.shape[0]
        thickness = text_config['thickness']
        font_scale = text_config['font_scale']
        space_between = text_config['space_between']
        x_pad, y_pad = text_config['pad']
        color = text_config['color']
            
        for (i, (k, v)) in enumerate(info_list):
            text = "{}:{}".format(k, v)
            cv2.putText(frame, text, (x_pad, ((i * space_between) + y_pad)), 
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
         
    def _draw_bottom_info_list(self, frame, info_list, text_config):
            video_width = frame.shape[1]
            video_height = frame.shape[0]
            thickness = text_config['thickness']
            font_scale = text_config['font_scale']
            space_between = text_config['space_between']
            x_pad, y_pad = text_config['pad']
            color = text_config['color']
            
            for (i, (k, v)) in enumerate(info_list):
                text = "{}:{}".format(k, v)
                cv2.putText(frame, text, (x_pad, video_height - ((i * space_between) + y_pad)), 
                            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)

def main():
    video_list_pcds_1 = [
        {
            'input_video': '../../test_dataset/pcds_3_back/2016_04_10_10_30_43BackColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/StandardSortTracker/pcds_3/logs/2016_04_10_10_30_43BackColor.txt',
            'output_video': '../../ProcessedVideos/2016_04_10_10_30_43BackColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_3_back/2016_04_07_16_26_47BackColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/StandardSortTracker/pcds_3/logs/2016_04_07_16_26_47BackColor.txt',
            'output_video': '../../ProcessedVideos/2016_04_07_16_26_47BackColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_1_front/2015_05_08_08_03_21FrontColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/CorrelationSortTracker/pcds_1/logs/2015_05_08_08_03_21FrontColor.txt',
            'output_video': '../../ProcessedVideos/2015_05_08_08_03_21FrontColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_0_front/2015_05_08_08_22_58FrontColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/CorrelationCentroidTracker/pcds_0/logs/2015_05_08_08_22_58FrontColor.txt',
            'output_video': '../../ProcessedVideos/2015_05_08_08_22_58FrontColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_2_back/2015_05_12_18_29_03BackColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/15_skip_frames/CorrelationCentroidTracker/pcds_2/logs/2015_05_12_18_29_03BackColor.txt',
            'output_video': '../../ProcessedVideos/2015_05_12_18_29_03BackColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_3_front/2015_05_08_13_06_10FrontColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/StandardSortTracker/pcds_3/logs/2015_05_08_13_06_10FrontColor.txt',
            'output_video': '../../ProcessedVideos/2015_05_08_13_06_10FrontColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_0_back/2015_05_12_16_11_37BackColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/15_skip_frames/CorrelationSortTracker/pcds_0/logs/2015_05_12_16_11_37BackColor.txt',
            'output_video': '../../ProcessedVideos/2015_05_12_16_11_37BackColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_3_front/2015_05_08_11_15_56FrontColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/5_skip_frames/StandardSortTracker/pcds_3/logs/2015_05_08_11_15_56FrontColor.txt',
            'output_video': '../../ProcessedVideos/2015_05_08_11_15_56FrontColor.avi',
        },
        {
            'input_video': '../../test_dataset/pcds_3_front/2015_05_12_14_13_30FrontColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/5_skip_frames/StandardSortTracker/pcds_3/logs/2015_05_12_14_13_30FrontColor.txt',
            'output_video': '../../ProcessedVideos/2015_05_12_14_13_30FrontColor.avi'
        },
    ]

    video_list_pcds_2 = [
        {
            'input_video': '../../test_dataset/pcds_1_front/2015_05_08_08_00_40FrontColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/StandardSortTracker/pcds_1/logs/2015_05_08_08_00_40FrontColor.txt',
            'output_video': '../../ProcessedVideos/new/2015_05_08_08_00_40FrontColor.avi'
        },
        {
            'input_video': '../../test_dataset/pcds_1_front/2016_04_10_18_45_20FrontColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/StandardSortTracker/pcds_1/logs/2016_04_10_18_45_20FrontColor.txt',
            'output_video': '../../ProcessedVideos/new/2016_04_10_18_45_20FrontColor.avi'
        },
    ]

    video_list_pcds_3 = [
        {
            'input_video': '../../test_dataset/pcds_1_back/2016_04_10_12_29_26BackColor.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/5_skip_frames/StandardSortTracker/pcds_1/logs/2016_04_10_12_29_26BackColor.txt',
            'output_video': '../../ProcessedVideos/new/2016_04_10_12_29_26BackColor.avi'
        }
    ]

    video_list_peruibe_1 = [
        {
            'input_video': '../../test_dataset/peruibe_4/0000000000000000-210713-072337-074337-000001000450.avi',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/CorrelationSortTracker/peruibe_4/logs/0000000000000000-210713-072337-074337-000001000450.txt',
            'output_video': '../../ProcessedVideos/0000000000000000-210713-072337-074337-000001000450.avi'
        },
        {
            'input_video': '../../test_dataset/peruibe_4/0000000000000000-210711-032004-034004-000001000470.mp4',
            'log_file': 'tests/runs/2023_12_03-03_53_37/5_skip_frames/CorrelationCentroidTracker/peruibe_4/logs/0000000000000000-210711-032004-034004-000001000470.txt',
            'output_video': '../../ProcessedVideos/0000000000000000-210711-032004-034004-000001000470.avi'
        },
    ]

    video_list_peruibe_2 = [
        {
            'input_video': '../../test_dataset/peruibe_4/0000000000000000-210712-032254-034115-000001000380.mp4',
            'log_file': 'tests/runs/2023_12_03-03_53_37/15_skip_frames/CorrelationSortTracker/peruibe_4/logs/0000000000000000-210712-032254-034115-000001000380.txt',
            'output_video': '../../ProcessedVideos/new/0000000000000000-210712-032254-034115-000001000380.avi'
        },
    ]

    video_list_mvia = [
        {
            'input_video': '../../test_dataset/mvia_5/C_O_G_2.mkv',
            'log_file': 'tests/runs/2023_12_03-03_53_37/10_skip_frames/StandardSortTracker/mvia_5/logs/C_O_G_2.txt',
            'output_video': '../../ProcessedVideos/C_O_G_2.avi'
        },
    ]
    
    dataset_name = 'peruibe'
    video_list  = video_list_peruibe_2

    for item in video_list:
        TrackingVideoGenerator.run(item['input_video'], item['output_video'], item['log_file'], 
                                   FPS_CONFIGS[dataset_name], INFO_TEXT_CONFIGS[dataset_name])

if __name__ == '__main__':
    main()