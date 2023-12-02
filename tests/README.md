# Testes

## Formato do JSON Resultados

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