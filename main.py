import cv2
import argparse
import multiprocessing
from lib.people_counter import PeopleCounter
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
    ap.add_argument("-c", "--confidence", type=float, default=0.8,
                    help="minimum probability to filter weak detections")
    ap.add_argument("-s", "--skip-frames", type=int, default=5,
                    help="# of skip frames between detections")
    args = vars(ap.parse_args())
    return args

def handle_up_down_event(data):
    print('Event: ', data)

if __name__ == '__main__':

    args = parse_arguments()

    videostream = cv2.VideoCapture(args["input"])

    people_counter = PeopleCounter(model_path=args["model"],
                                confidence=args["confidence"],
                                num_threads=multiprocessing.cpu_count(),
                                videostream=videostream,
                                skip_frames=args["skip_frames"],
                                output_file=args["output"],
                                ct_max_distance=60,
                                ct_max_disappeared=30,
                                up_down_handler=(UpDownEventHandler(handle_up_down_event, 5))
                                )

    print("Iniciando contagem...")
    people_counter.start_counting()
    print("Contagem finalizada...")
    print("FPS: ", people_counter.get_current_fps())

"""
TODO

- update_open_event -> increment logic
- ajuste do centroid tracker

- add sort tracker
- entrance_border
- entrance_direction (TOP_TO_BOTTOM | BOTTOM_TO_TOP)
- adaptar para 0 skip_frames
- get stream of rpi camera
"""

"""
correlation 
	update: img_rgb, boxes

centroid
	update: boxes
	ct_max_disappeared =>
	ct_max_distance    =>

sort
	update: boxes, scores
	max_age=1  	  => Maximum number of frames to keep alive a track without associated detections.	
	min_hits=3	  => Minimum number of associated detections before track is initialised.
	iou_threshold=0.3 => Minimum IOU for match.

ct_max_disappeared = ct_max_distance
iou_threshold é uma versao mais inteligente do ct_max_distance

Detection(box, score, label)

correlation: 
	update: Detections, img_rgb -> 
	

centroid: 
	update: Detections -> 

sort: 
	update: Detections -> 

- viabilidade de ignorar score (aparentemente o sort nao usa score)

-------

PeopleCounter -> recebe objeto tracker como parametro

Tracker Generico (Super Class) (Abstract class)
    - init() -> 
    - update()
        - no skip_frames
            (box, img_rgb)
        - skip_frames
    - (sort / centroid) tracker

update
    - centroid
        - y_new_pos(box)[]: retorna os objetos considerando new, matched, unmatched objects
        - n_new_pos()[]: retorna os objetos ja rastreados (sem estimativa de deslocamento)
        - return: id, centroid
    - sort
        - y_new_pos(box, score)[]: retorna os objetos considerando new, matched, unmatched objects
        - n_new_pos()[]: retorna os objetos ja rastreados (com estimativa de deslocamento)
        - return: id, box, score 
    - correlation
        - y_new_pos(box, img_rgb)[]: start tracker do objeto
        - n_new_pos(img_rgb)[]: retorna estimativa da nova posticao do objeto
        - return: box

template_method(GenericDectection[]) 
    ret = update_abstract_method(GenericDectection[])
    # ret typeof (GenericNewPos[])

para implementacos com correlation tracker, as ConcreteClasses devem armazenar trackers de correlacao durante os casos de n_new_pos
"""