#!/usr/bin/env bash

#python3 -m imet.train \
#    --dataset-path data/imet/ \
#    --config-path imet/config/1cyc_pad_aspect.yml \
#    --experiment-path tf_log/imet/1cyc-pad-aspect \
#    --fold 1

python3 -m imet.train \
    --dataset-path data/imet/ \
    --config-path imet/config/1cyc_pad_lsep.yml \
    --experiment-path tf_log/imet/1cyc-pad-lsep \
    --fold 1

python3 -m imet.train \
    --dataset-path data/imet/ \
    --config-path imet/config/1cyc_crop.yml \
    --experiment-path tf_log/imet/1cyc-crop \
    --fold 1

python3 -m imet.train \
    --dataset-path data/imet/ \
    --config-path imet/config/1cyc_resize.yml \
    --experiment-path tf_log/imet/1cyc-resize \
    --fold 1

python3 -m imet.train \
    --dataset-path data/imet/ \
    --config-path imet/config/plat_pad.yml \
    --experiment-path tf_log/imet/plat-pad \
    --fold 1

