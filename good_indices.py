# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Preprocessing code for the protein-ligand complex.
"""

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse


def save_good_indices(dataset_source, output_path, dataset_name):
    if dataset_source.endswith('.tsv'):
       df = pd.read_csv(dataset_source, sep='\t')
    elif dataset_source.endswith('.csv'):
       df = pd.read_csv(dataset_source)

    df = df.drop(columns={'Unnamed: 0'})
    df_train = df.loc[df['type'] == 'training'].reset_index()
    df_val = df.loc[df['type'] == 'validation'].reset_index()
    df_test = df.loc[df['type'] == 'test'].reset_index()

    train_idx = df_train.query('rmsd < 1.5').index
    val_idx = df_val.query('rmsd < 1.5').index
    test_idx = df_test.query('rmsd < 1.5').index

    print('Train len:', len(train_idx))
    print('Val len:', len(val_idx))
    print('Test len:', len(test_idx))

    np.save(os.path.join(output_path, dataset_name + '_idx_train.npy'), train_idx, allow_pickle=False)
    np.save(os.path.join(output_path, dataset_name + '_idx_val.npy'), val_idx, allow_pickle=False)
    np.save(os.path.join(output_path, dataset_name + '_idx_test.npy'), test_idx, allow_pickle=False)


def load_good_indices(data_path, dataset_name):
    train_idx = None
    val_idx = None
    test_idx = None

    train_path = os.path.join(data_path, dataset_name + '_idx_train.npy')
    val_path = os.path.join(data_path, dataset_name + '_idx_val.npy')
    test_path = os.path.join(data_path, dataset_name + '_idx_test.npy')

    if os.path.exists(train_path):
        train_idx = np.load(train_path)
    if os.path.exists(val_path):
        val_idx = np.load(val_path)
    if os.path.exists(test_path):
        test_idx = np.load(test_path)

    return train_idx, val_idx, test_idx


def load_indices(df_path, good=False):
    train_path = df_path + '_training.csv'
    val_path = df_path + '_validation.csv'
    test_path = df_path + '_test.csv'

    if good:
        train_idx = pd.read_csv(train_path).query('rmsd < 1.5').index
        val_idx = pd.read_csv(val_path).query('rmsd < 1.5').index
        test_idx = pd.read_csv(test_path).query('rmsd < 1.5').index
    else:
        train_idx = pd.read_csv(train_path).index
        val_idx = pd.read_csv(val_path).index
        test_idx = pd.read_csv(test_path).index

    return train_idx, val_idx, test_idx


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--output_path', type=str, default='./data/')
    parser.add_argument('--dataset_name', type=str, default='63k')
    args = parser.parse_args()
    save_good_indices(args.dataset_file, args.output_path, args.dataset_name)
