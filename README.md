# Algoritmo de Contagem Passageiros

## Dependências de SO

- Python 3.7.x (3.7.16) 
- virtualenv
- libboost-all-dev
- libgtk-3-dev
- build-essential
- cmake

## Instalação

Crie um ambiente python com o virtualenv:
```
virtualenv venv --python="/usr/bin/python3.7"
```

Ative o ambiente criado:
```
source venv/bin/activate
```

Instale as dependências:
```
pip install -r requirements.txt
```

## Execução 

### Parâmetros

|Parâmetro|Descrição|Obrigatório|Default|
|---------|---------|-----------|-------|
|--model|Caminho do modelo .tflite de dectecção de pessoas.|Sim|-|
|--conf-thresh|Limite mínimo de confiança para as detecções realizadas.|Não|0.9|
|--skip-frames|Quantidade de frames que devem ser ignorados antes de realizar uma nova detecção.|Não|15|
|--input|Caminho do vídeo de entrada ou número do dispositivo(câmera) conectado.|Sim|-|
|--output|Caminho do vídeo com os resultados de processamento. |Não|-|
|--log-file|Caminho do arquivo de logs de processamento.|Não|./logs/\<data-hora\>.txt|

### Comando

Com o ambiente ativado, execute o comando a seguir para iniciar o programa de contagem:
```
python main.py --model <value> --conf-thresh <value> --skip-frames <value> --input <value> --output <value> --log-file <value> 
```

## Profiling

```
test_ python -m cProfile -s time <prog.py>
```

## Algoritmos de Rastreamento de Objetos

### Centroid Tracker

#### Parâmetros

|Parâmetro|Descrição|
|---------|---------|
|max_disappeared|Número máximo de frames para manter vivo um objeto rastreado sem detecções associadas.|
|max_distance|Distância máxima de match utilizado na etapa de associação de objetos.|

### SORT Tracker

#### Parâmetros

|Parâmetro|Descrição|
|---------|---------|
|max_age|Número máximo de frames para manter vivo um objeto rastreado sem detecções associadas.|
|min_hits|Número mínimo de detecções associadas antes da inicialização do rastreamento.|
|iou_threshold|IOU mínimo de match utilizado na etapa de associação de objetos.|

Observações:
- O max_disappeared do Centroid Tracker equivale ao max_age do SORT Tracker.
- O iou_threshold do SORT é uma versão mais sofisticada do max_distance do Centroid Tracker.

## Formato do Arquivo de Logs

```
<tracker_info>, <skip_frames>, <num_threads>, <conf_thresh>, <entrance_border>, <entrance_direction>
<frame_number> <fps> <entering_number> <exiting_number>, <object_id> <xmin> <ymin> <xmax> <ymax>, ...
...
<frame_number> <fps> <entering_number> <exiting_number>, <object_id> <xmin> <ymin> <xmax> <ymax>, ...
```