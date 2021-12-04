#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

TRAIN_URL="https://www.dropbox.com/s/cipluc1qyrtx0hi/pdbsmall_train.pkl"
VAL_URL="https://www.dropbox.com/s/zthjn1g9xutr5f6/pdbsmall_val.pkl"
TEST_URL="https://www.dropbox.com/s/j8zpii5d6e3l1wr/pdbsmall_test.pkl"

wget -P "../data" $TRAIN_URL
wget -P "../data" $VAL_URL
wget -P "../data" $TEST_URL
