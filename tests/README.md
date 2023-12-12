# Testes

Os relatórios com os resultados do teste realizado na Raspberry Pi 4 estão disponíveis na seguinte pasta: 

- https://github.com/TCC-GustavoNr/TCC-Passenger-Counting-Algorithm/tree/main/tests/runs/2023_12_03-03_53_37

Consulte as seções a seguir para compreender a forma que os resultados foram estruturados.

## Formato da Pasta de Resultados

```
runs/
    run_folder/
        x_skip_frames/
            trackerconfig/
                datasetname/
                    logs/
                    report.txt    
                report.txt
        results.json
```

Foram gerados dois tipos de relatórios para cada configuração de rastreador de objetos, sendo eles:

- A nível de rastreador: contém o desempenho do rastreador para todo o dataset de teste.
    - Exemplo: x_skip_frames/trackerconfig/report.txt
- A nível de dataset: contém o desempenho do rastreador para um dataset específico.
    - Exemplo: x_skip_frames/trackerconfig/datasetname/report.txt

Os arquivos "report.txt" contam com as seguintes métricas de desempenho: FPS médio, Precisão e Recall.

## Formato do JSON Resultados

O arquivo "results.json" contempla o relatório completo do teste e possui com o seguinte formato:

```json
{
    "x_skip_frames": {
        "trackerconfig": {
            "global": {
                "avg_fps": 0,
                "std_fps": 0,
                "precision": 0, 
                "recall": 0,
                "tp": 0, "fp": 0, "fn": 0,
                "entering": {"precision": 0, "recall": 0},
                "exiting": {"precision": 0, "recall": 0},
            },
            "cumulative": {
                "fps": [],
                "entering": {"tp": 0, "fp": 0, "fn": 0},
                "exiting": {"tp": 0, "fp": 0, "fn": 0},
            },
            "datasets": {
                "datasetname": {
                    "global": {
                        "avg_fps": 0,
                        "std_fps": 0,
                        "precision": 0, 
                        "recall": 0,
                        "tp": 0, "fp": 0, "fn": 0,
                        "entering": {"precision": 0, "recall": 0},
                        "exiting": {"precision": 0, "recall": 0},
                    },
                    "cumulative": {
                        "fps": [],
                        "entering": {"tp": 0, "fp": 0, "fn": 0},
                        "exiting": {"tp": 0, "fp": 0, "fn": 0},
                    },
                    "videos": [{
                        "avg_fps": 0,
                        "video_path": "",
                        "configs": {},
                        "prediction": {"entering": 0, "exiting": 0},
                        "ground_truth": {"entering": 0, "exiting": 0},
                    }]
                }
            }
        }
    }
}
```
