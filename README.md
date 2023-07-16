# TCC1 - Algoritmo de Contagem Passageiros

Versão experimental do algoritmo de contagem de passageiros.

O algoritmo foi baseado em abordagem que combina detecção de pessoas (Single Shot Detector) e rastreamento de pessoas (Centroid Tracker). Além disso, utiliza como base as implementações desenvolvidas por Rosebrock (2021) e Subhakar (2022) na linguagem de programação Python.

## Rodando

- virtualenv venv && source venv/bin/activate 

- virtualenv venv --python="/usr/bin/python3.7" && source venv/bin/activate

- pip install -r requirements.txt

- python main.py --model mobilenet_ssd/detect.tflite --input videos/inputs/cut_sample_1_4.mp4 --output videos/outputs/output.mp4

- python main.py --model mobilenet_ssd/detect.tflite --skip-frames 2 --input videos/inputs/sample_2.mkv --output videos/outputs/output_4.mp4

- python main.py --model mobilenet_ssd/detect.tflite --input ../videos/inputs/cut_sample_1_4.mp4 --output ../videos/outputs/output_8.mp4

- python main.py --model mobilenet_ssd/v2/detect.tflite --input ../videos/inputs/pcds_front_2.avi --output ../videos/outputs/output_25.mp4

- python -m cProfile -s time <prog.py>

- Python 3.7.16

## Referências

- ROSEBROCK, A. Opencv people counter. 2021. Acessado em 25 de janeiro de 2023. Disponível em: <https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/>.

- SUBHAKAR, S. People counting in real time. 2022. Acessado em 25 de janeiro de 2023. Disponível em: <https://github.com/saimj7/People-Counting-in-Real-Time>. 

## Uso

centroid
	update: boxes
	ct_max_disappeared =>
	ct_max_distance    =>

sort
	update: boxes, scores
	max_age=1  	  => Maximum number of frames to keep alive a track without associated detections.	
	min_hits=3	  => Minimum number of associated detections before track is initialised.
	iou_threshold=0.3 => Minimum IOU for match.

ct_max_disappeared = max_age
iou_threshold é uma versao mais inteligente do ct_max_distance