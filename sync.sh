#!/usr/bin/env bash

#HOST=vshmyhlo@192.168.0.103
HOST=vshmyhlo@77.120.246.182

rsync -avhHPx ./*.py ${HOST}:code/kaggle/
rsync -avhHPx ./imet/ ${HOST}:code/kaggle/imet/
rsync -avhHPx ./frees/ ${HOST}:code/kaggle/frees/
