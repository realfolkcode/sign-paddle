#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

./download.sh pdbsmall
python ../train.py --cuda -1 --model_dir ../models --dataset pdbsmall --cut_dist 5 --num_angle 6 > ../logs/train_log
