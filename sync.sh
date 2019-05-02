#!/usr/bin/env bash

HOST=vshmyhlo@192.168.0.103

rsync -avhHPx ./*.py ${HOST}:code/kaggle/
rsync -avhHPx ./imet/ ${HOST}:code/kaggle/imet/
rsync -avhHPx ./frees/ ${HOST}:code/kaggle/frees/
rsync -avhHPx ${HOST}:code/kaggle/submission.csv ./submission.csv
