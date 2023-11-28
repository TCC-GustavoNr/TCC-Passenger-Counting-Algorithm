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

<frame_number> <fps> <entering-number> <exiting-number> <skip_frames> <entrance_border> <conf_thresh>
<object_id> <xmin> <ymin> <xmax> <ymax> <conf>
...
<object_id> <xmin> <ymin> <xmax> <ymax> <conf>

--------------------

<frame_number> <fps> <entering-number> <exiting-number> <skip_frames> <entrance_border> <conf_thresh>, <object_id> <xmin> <ymin> <xmax> <ymax> <conf>, ...


=================================
PARAMETROS

# PDCS

- skip_frames=10
- front:  entrance_border=0.65


# PERUIBE

- skip_frames=10
- entrance_border=0.50

# MVIA 

- skip_frames=10
- entrance_border=0.50