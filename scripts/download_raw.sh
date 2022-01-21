#!/bin/bash

PARENT_PATH=$( cd "$(dirname "${BASH_SOURCE[0]}")" ; pwd -P )

cd "$PARENT_PATH"

PDBBIND_URL="https://www.dropbox.com/sh/2uih3c6fq37qfli/AAAO_w7sZJE6D0GsF-iWoEOGa/PDBbind_dataset.tar.gz"

DATASET_URL="https://www.dropbox.com/s/f4xq6bb6bci457t/67k_docked.zip"

TARGET_URL="https://www.dropbox.com/s/4sx8vtrhwjxunwj/67k-energy_rmsd.tsv"

wget -P "../data" $PDBBIND_URL
wget -P "../data" $DATASET_URL
wget -P "../data" $TARGET_URL
tar -xf "../data/PDBbind_dataset.tar.gz" -C "../data"
unzip -aq "../data/67k_docked.zip" -d "../data"
