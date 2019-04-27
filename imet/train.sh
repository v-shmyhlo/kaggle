#!/usr/bin/env bash

python3 -m imet.train \
    --image-size 256 \
    --batch-size 60 \
    --epochs 20 \
    --aug pad \
    --opt adam \
    --anneal linear \
    --weight-decay 2e-4 \
    --experiment-path tf_log/imet/256-e20-pad-adam-lin-wd2e-4 \
    --dataset-path data/imet \
    --fold 1

python3 -m imet.train \
    --image-size 256 \
    --batch-size 60 \
    --epochs 20 \
    --aug pad \
    --opt adam \
    --anneal linear \
    --weight-decay 4e-4 \
    --experiment-path tf_log/imet/256-e20-pad-adam-lin-wd4e-4 \
    --dataset-path data/imet \
    --fold 1

python3 -m imet.train \
    --image-size 256 \
    --batch-size 60 \
    --epochs 10 \
    --aug pad \
    --opt adam \
    --anneal linear \
    --weight-decay 2e-4 \
    --experiment-path tf_log/imet/256-e10-pad-adam-lin-wd2e-4 \
    --dataset-path data/imet \
    --fold 1

python3 -m imet.train \
    --image-size 256 \
    --batch-size 60 \
    --epochs 10 \
    --aug pad \
    --opt adam \
    --anneal cosine \
    --weight-decay 2e-4 \
    --experiment-path tf_log/imet/256-e10-pad-adam-cos-wd2e-4 \
    --dataset-path data/imet \
    --fold 1
