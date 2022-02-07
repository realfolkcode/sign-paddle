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
Training process code for Structure-aware Interactive Graph Neural Networks (SIGN).
"""
import os
import time
import math
import argparse
import random
import numpy as np
import pickle5 as pickle

import paddle
import paddle.nn.functional as F
from model import SIGN
from tqdm import tqdm

def count_interactions(data_path, dataset, cut_dist, num_angles):
    c = 0
    idx_lst = []
    graph_prefix = f'{dataset}_{int(cut_dist)}_{num_angles}_pgl_graph_'
    for filename in os.listdir(data_path):
        if filename.startswith(graph_prefix):
            with open(os.path.join(data_path, filename), 'rb') as f:
                _, global_feat, _ = pickle.load(f)
                if np.sum(global_feat[2]) == 0:
                    c += 1
                else:
                    idx = filename[len(graph_prefix):-4]
                    idx_lst.append(idx)
    np.savetxt(os.path.join(data_path, 'interactions.txt'), np.array(idx_lst).reshape(-1))
    print(c)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--dataset', type=str, default='pdbbind2016')
    parser.add_argument('--cut_dist', type=float, default=5.)
    parser.add_argument('--num_angle', type=int, default=6)

    args = parser.parse_args()

    count_interactions(args.data_dir, "%s_train" % args.dataset, args.cut_dist, args.num_angle)
