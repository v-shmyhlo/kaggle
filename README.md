# Recursion Cellular Image Classification

### сгенерировать статистику для нормализации:
```bash
python3 -m cells.build_experiment_stats \
    --dataset-path ../../data/cells
```

### запустить тренировку первого фолда:
```bash
CUDA_VISIBLE_DEVICES=1 python3 -m cells.train \
    --dataset-path ../../data/cells \
    --config-path cells/config/effnet_b0_512_la_ewa.yaml \
    --experiment-path tf_log/cells/hell-world \
    --fold 1
```

### запустить тренировку c псевдолейблингом из чекпоинта:
```bash
CUDA_VISIBLE_DEVICES=1 python3 -m cells.train_pl \
    --dataset-path ../../data/cells \
    --config-path cells/config/effnet_b0_512_la_ewa_pl.yaml \
    --experiment-path tf_log/cells/hell-world-pl \
    --pl-path tf_log/cells/hell-world \
    --restore-path tf_log/cells/hell-world \
    --fold 1
```

### ну и теперь сгенерить предикшены используя хак:
```bash
CUDA_VISIBLE_DEVICES=1 python3 -m cells.train_hack \
    --dataset-path ../../data/cells \
    --config-path cells/config/effnet_b0_512_la_ewa.yaml \
    --experiment-path tf_log/cells/hell-world \
    --infer
```
