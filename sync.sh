#!/usr/bin/env bash

HOST=vshmyhlo@192.168.1.9
# HOST=vshmyhlo@93.73.3.214

rsync -avhHPx ./kaggle.yml ${HOST}:.config/tmuxinator/kaggle.yml
rsync -avhHPx "${HOST}:.kaggle/kaggle.json" ~/.kaggle/kaggle.json

rsync -avhHPx ./*.{py,txt} ${HOST}:code/kaggle/

for dir in detection
do
    rsync -avhHPx ./${dir}/ ${HOST}:code/kaggle/${dir}/
done
