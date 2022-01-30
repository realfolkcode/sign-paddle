#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

TRAIN_URL="1g7J2BBWvsgLD-BhCP7wjomnH2GCNJXZ4"
VAL_URL="18lOrRBzWoFyjCWdKMQGnta4n9hxZfh3Q"
TEST_URL="12wK5f3Fvz-yxb667rinV28ibgcuSQve6"

gdown --id $TRAIN_URL --output "../data/63k_train.pkl"
gdown --id $VAL_URL --output "../data/63k_val.pkl"
gdown --id $TEST_URL --output "../data/63k_test.pkl"
