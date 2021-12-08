#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

./download.sh pdbbind_2016
python ../train.py --cuda 4 --model_dir ../models --data_dir ../data --dataset pdbbind_2016 --cut_dist 5 --num_angle 6 > ../logs/train_log
