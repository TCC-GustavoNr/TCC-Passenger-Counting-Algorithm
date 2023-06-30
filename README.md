# TCC1 - Algoritmo de Contagem Passageiros

Versão experimental do algoritmo de contagem de passageiros.

O algoritmo foi baseado em abordagem que combina detecção de pessoas (Single Shot Detector) e rastreamento de pessoas (Centroid Tracker). Além disso, utiliza como base as implementações desenvolvidas por Rosebrock (2021) e Subhakar (2022) na linguagem de programação Python.

## Referências

- ROSEBROCK, A. Opencv people counter. 2021. Acessado em 25 de janeiro de 2023. Disponível em: <https://www.pyimagesearch.com/2018/08/13/opencv-people-counter/>.

- SUBHAKAR, S. People counting in real time. 2022. Acessado em 25 de janeiro de 2023. Disponível em: <https://github.com/saimj7/People-Counting-in-Real-Time>. 

## Exec

- virtualenv venv && source venv/bin/activate 

- virtualenv venv --python="/usr/bin/python3.7" && source venv/bin/activate

- python pip install -r requirements.txt

- python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input videos/inputs/cut_sample_1_4.mp4 --output videos/outputs/output.mp4

- python main.py --prototxt mobilenet_ssd/MobileNetSSD_deploy.prototxt --model mobilenet_ssd/MobileNetSSD_deploy.caffemodel --input images/inputs/p3_3.png --output images/outputs/output.png

- python main.py --model mobilenet_ssd/detect.tflite --input videos/inputs/cut_sample_1_4.mp4 --output videos/outputs/output.mp4

- python main.py --model mobilenet_ssd/detect.tflite --skip-frames 2 --input videos/inputs/sample_2.mkv --output videos/outputs/output_4.mp4

- python -m cProfile -s time <prog.py>

- Python 3.7.16
## Req

schedule==1.1.0
numpy
argparse==1.4.0
imutils==0.5.4
dlib==19.18.0
opencv-python==4.5.5.64
scipy
cmake==3.22.5

## Update venv

https://www.pythonanywhere.com/forums/topic/11681/