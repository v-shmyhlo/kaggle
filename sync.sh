#!/usr/bin/env bash

HOST=vshmyhlo@77.120.246.182

rsync -avhHPx ./kaggle.yml ${HOST}:.config/tmuxinator/kaggle.yml
rsync -avhHPx ./requirements.txt ${HOST}:code/kaggle/requirements.txt
rsync -avhHPx ./*.py ${HOST}:code/kaggle/
rsync -avhHPx ./imet/ ${HOST}:code/kaggle/imet/
rsync -avhHPx ./frees/ ${HOST}:code/kaggle/frees/
rsync -avhHPx ./mol/ ${HOST}:code/kaggle/mol/
rsync -avhHPx ./detection/ ${HOST}:code/kaggle/detection/
rsync -avhHPx ./test/ ${HOST}:code/kaggle/test/
