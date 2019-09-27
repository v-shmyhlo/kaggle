#!/usr/bin/env bash

HOST=vshmyhlo@77.120.246.182

rsync -avhHPx ./kaggle.yml ${HOST}:.config/tmuxinator/kaggle.yml
rsync -avhHPx ./requirements.txt ${HOST}:code/kaggle/requirements.txt
rsync -avhHPx ./*.py ${HOST}:code/kaggle/
rsync -avhHPx "${HOST}:.kaggle/kaggle.json" ~/.kaggle/kaggle.json

rsync -avhHPx ./pl-data/ ${HOST}:code/kaggle/pl-data/
# rsync -avhHPx "${HOST}:code/kaggle/*.csv" ./csv/
# rsync -avhHPx "${HOST}:code/kaggle/*.pth" ./my-pth/
rsync -avhHPx "${HOST}:code/kaggle/*.pth" ./my-pl-pth/
# rsync -avhHPx "${HOST}:code/kaggle/tf_log/cells/tmp-512-progres-crop-norm-la-pl-restore-2/" ./solution-tta/
# rsync -avhHPx "${HOST}:code/kaggle/tf_log/cells/tmp-512-progres-crop-norm-la/" ./solution-nopl-tho/

for path in cells stal imet frees mol classification segmentation detection test
do
    rsync -avhHPx ./${path}/ ${HOST}:code/kaggle/${path}/
done
