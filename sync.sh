#!/usr/bin/env bash

rsync -avhHPx ./*.py vshmyhlo@192.168.0.105:code/kaggle/
rsync -avhHPx ./imet/ vshmyhlo@192.168.0.105:code/kaggle/imet/
rsync -avhHPx vshmyhlo@192.168.0.105:code/kaggle/submission.csv ./submission.csv
