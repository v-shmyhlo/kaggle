#!/usr/bin/env bash

#HOST=vshmyhlo@192.168.0.103
HOST=vshmyhlo@77.120.246.182

rsync -avhHPx ./*.py ${HOST}:code/kaggle/
rsync -avhHPx ./imet/ ${HOST}:code/kaggle/imet/
rsync -avhHPx ./frees/ ${HOST}:code/kaggle/frees/
rsync -avhHPx ./mol/ ${HOST}:code/kaggle/mol/
rsync -avhHPx ./retinanet/ ${HOST}:code/kaggle/retinanet/

# rsync -avhHPx ./data/mol/ ${HOST}:code/kaggle/data/mol/

#rsync -avhHPx \
#    ${HOST}:code/kaggle/tf_log/frees/1cyc90-crop15-adam8e-4-rsplit-cancel-canspecaugT50-mix-fx/model_*.pth \
#    ./weights/1/
#
#rsync -avhHPx \
#    ${HOST}:code/kaggle/tf_log/frees/1cyc120-crop15-adam8e-4-rsplit-cancel-canspecaugT50-mix-nonoisy-fx/model_*.pth \
#    ./weights/2/
#
#rsync -avhHPx \
#    ${HOST}:code/kaggle/tf_log/frees/1cyc90-crop15-adam8e-4-rsplit-cancel-canspecaugT50-mix-noisy1k/model_*.pth \
#    ./weights/3/
#
#rsync -avhHPx \
#    ${HOST}:code/kaggle/tf_log/frees/1cyc90-crop15-adam8e-4-rsplit-cancel-canspecaugT50-mix-fx-newnoisy/model_*.pth \
#    ./weights/4/
