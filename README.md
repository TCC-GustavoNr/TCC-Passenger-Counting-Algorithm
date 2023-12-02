# Algoritmo de Contagem Passageiros

## Rodando

- virtualenv venv --python="/usr/bin/python3.7" && source venv/bin/activate

- pip install -r requirements.txt

- virtualenv venv && source venv/bin/activate 

-----------------

- python main.py --model mobilenet_ssd/detect.tflite --input videos/inputs/cut_sample_1_4.mp4 --output videos/outputs/output.mp4

- python main.py --model mobilenet_ssd/detect.tflite --skip-frames 2 --input videos/inputs/sample_2.mkv --output videos/outputs/output_4.mp4

- python main.py --model mobilenet_ssd/detect.tflite --input ../videos/inputs/cut_sample_1_4.mp4 --output ../videos/outputs/output_8.mp4

- python main.py --model mobilenet_ssd/v2/detect.tflite --input ../videos/inputs/pcds_front_2.avi --output ../videos/outputs/output_25.mp4

- python main.py --skip-frames 10 --model mobilenet_ssd/v2/detect.tflite --input ../../test_dataset/1/2016_04_10_18_45_20FrontColor.avi --output ./2016_04_10_18_45_20FrontColor_3.mp4

- python main.py --skip-frames 10 --model mobilenet_ssd/v2/detect.tflite --input ../../test_dataset/4/0000000000000000-210713-042508-043003-000001000190.avi --output ./0000000000000000-210713-042508-043003-000001000190_1.mp4

- python main.py --skip-frames 10 --model mobilenet_ssd/v2/detect.tflite --input ../../test_dataset/5/C_O_G_2.mkv --output ./C_O_G_2_1.mp4

- python main.py --skip-frames 10 --model mobilenet_ssd/v2/detect.tflite --input ../../test_dataset/1/2016_04_10_18_45_20FrontColor.avi

- python main.py --skip-frames 10 --model mobilenet_ssd/v2/detect.tflite --input ../test_dataset/1/2016_04_10_18_45_20FrontColor.avi

- python main.py --skip-frames 10 --model mobilenet_ssd/v2/detect.tflite --input ../../test_dataset/pcds_0_back/2016_04_09_21_50_15BackColor.avi --output ./2016_04_09_21_50_15BackColor.avi

python main.py --skip-frames 10 --model mobilenet_ssd/v2/detect.tflite --input ../../test_dataset/pcds_1_front/2016_04_10_18_45_20FrontColor.avi --output ./2016_04_10_18_45_20FrontColor.avi

python main.py --skip-frames 1 --model mobilenet_ssd/v2/detect.tflite --input ../../test_dataset/pcds_0_front/2015_05_08_08_22_58FrontColor.avi --output ./2015_05_08_08_22_58FrontColor.avi

test_ python -m cProfile -s time <prog.py>

- Python 3.7.16 

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

# Tempos

## Notebook 

PCDS
AVG FPS: 26.011605415860735
AVG FPS: 25.550612508059316

AVG FPS: 22.87878787878788
AVG FPS: 27.534493874919406
AVG FPS: 24.777562862669246

PERUIBE
AVG FPS: 111.73242604881341

MVIA
AVG FPS: 48.20506965060516

# Skips

0 5 10 15 20 25 30

