#!/usr/bin/env bash

HOST=vshmyhlo@77.120.246.182

rsync -avhHPx ./kaggle.yml ${HOST}:.config/tmuxinator/kaggle.yml
rsync -avhHPx ./requirements.txt ${HOST}:code/kaggle/requirements.txt
rsync -avhHPx ./*.py ${HOST}:code/kaggle/
# rsync -avhHPx "${HOST}:code/kaggle/*.csv" ./csv/
rsync -avhHPx "${HOST}:code/kaggle/tf_log/cells/tmp-512-progres-crop-norm-la-pl-restore-2/" ./solution/
rsync -avhHPx "${HOST}:code/kaggle/tf_log/cells/tmp-512-progres-crop-norm-la/" ./solution-nopl/

for path in cells stal imet frees mol classification segmentation detection test
do
    rsync -avhHPx ./${path}/ ${HOST}:code/kaggle/${path}/
done
