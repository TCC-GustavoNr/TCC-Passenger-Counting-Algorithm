TODO

- update_open_event -> increment logic

- add generic tracker (ok)
- entrance_border (ok)
- entrance_direction (TOP_TO_BOTTOM | BOTTOM_TO_TOP) (ok)
- adaptar para 0 skip_frames (ok)
- get stream of rpi camera
- tratar mesmo objeto sobe e desce 
- log das detecoes/rastreamento em arquivo
- parametro --view 

=================================
Log
Para cada frame:

<tracker_info>, <skip_frames>, <num_threads>, <conf_thresh>, <entrance_border>, <entrance_direction>
<frame_number> <fps> <entering_number> <exiting_number>, <object_id> <xmin> <ymin> <xmax> <ymax>, ...
...
<frame_number> <fps> <entering_number> <exiting_number>, <object_id> <xmin> <ymin> <xmax> <ymax>, ...

=================================
Skips

- skip_frames
- skip_detect
- skip_ctrack

=================================
PARAMETROS

# PDCS

## Front
- skip_frames=10
- entrance_border=0.65
- tracker = StandardSortTracker(max_age=60, min_hits=1, iou_threshold=0.3)

## Back


# PERUIBE

- skip_frames=10
- entrance_border=0.50

# MVIA 

- skip_frames=10
- entrance_border=0.50
- tracker = StandardSortTracker(max_age=30, min_hits=1, iou_threshold=0.2)