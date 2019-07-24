#!/usr/bin/env bash

HOST=vshmyhlo@77.120.246.182

rsync -avhHPx ./kaggle.yml ${HOST}:.config/tmuxinator/kaggle.yml
rsync -avhHPx ./requirements.txt ${HOST}:code/kaggle/requirements.txt
rsync -avhHPx ./*.py ${HOST}:code/kaggle/
rsync -avhHPx ${HOST}:code/kaggle/fig.png ./

for p in imet cells frees mol classification segmentation detection test
do
    rsync -avhHPx ./${p}/ ${HOST}:code/kaggle/${p}/
done
