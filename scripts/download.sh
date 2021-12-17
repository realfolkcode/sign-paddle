#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

DATASET=$1

if [ "$DATASET" == "pdbsmall" ]; then
	TRAIN_URL="https://www.dropbox.com/s/cipluc1qyrtx0hi/pdbsmall_train.pkl"
	VAL_URL="https://www.dropbox.com/s/zthjn1g9xutr5f6/pdbsmall_val.pkl"
	TEST_URL="https://www.dropbox.com/s/j8zpii5d6e3l1wr/pdbsmall_test.pkl"
elif [ "$DATASET" == "pdbbind2016" ]; then
	TRAIN_URL="https://www.dropbox.com/sh/68vc7j5cvqo4p39/AAB9fWY5sYoiTmEzHU4iXFPfa/pdbbind2016_train.pkl"
	VAL_URL="https://www.dropbox.com/sh/68vc7j5cvqo4p39/AAB5RuvgkQI1q4Hnl-C0J6mOa/pdbbind2016_val.pkl"
	TEST_URL="https://www.dropbox.com/sh/68vc7j5cvqo4p39/AAAJkCL1mrSyi6HqwhdW79bNa/pdbbind2016_test.pkl"
elif [ "$DATASET" == "pdb100" ]; then
	TRAIN_URL="https://www.dropbox.com/s/k8i3ut2oj6nvdzq/pdb100_train.pkl"
	VAL_URL="https://www.dropbox.com/s/84lpa1o6svj92uc/pdb100_val.pkl"
	TEST_URL="https://www.dropbox.com/s/wktpexd5iuk52ck/pdb100_test.pkl"
fi

wget -P "../data" $TRAIN_URL
wget -P "../data" $VAL_URL
wget -P "../data" $TEST_URL
