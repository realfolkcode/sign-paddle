#!/bin/bash

./download.sh pdbsmall
python train.py --cuda -1 --model_dir ./models --dataset pdbsmall --cut_dist 5 --num_angle 6 > ./logs/train_log
