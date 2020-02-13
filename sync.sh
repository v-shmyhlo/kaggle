#!/usr/bin/env bash


rsync -avhHPx ./kaggle.yml ${DEVBOX}:.config/tmuxinator/kaggle.yml
rsync -avhHPx "${DEVBOX}:.kaggle/kaggle.json" ~/.kaggle/kaggle.json
rsync -avhHPx ./*.{py,txt} ${DEVBOX}:code/kaggle/

for dir in beng
do
    rsync -avhHPx ./${dir}/ ${DEVBOX}:code/kaggle/${dir}/
done
