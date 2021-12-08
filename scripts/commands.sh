#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

./download.sh pdbbind2016
python ../train.py --cuda 0 --model_dir ../models --data_dir ../data --dataset pdbbind2016 --cut_dist 5 --num_angle 6 > ../logs/train_log
