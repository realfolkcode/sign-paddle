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
Dataset code for protein-ligand complexe interaction graph construction.
"""

import os
import numpy as np
import paddle
import pgl
import pickle5 as pickle
from pgl.utils.data import Dataset as BaseDataset
from pgl.utils.data import Dataloader
from scipy.spatial import distance
from scipy.sparse import coo_matrix
from utils import cos_formula
from tqdm import tqdm
import pandas as pd

prot_atom_ids = [6, 7, 8, 16]
drug_atom_ids = [6, 7, 8, 9, 15, 16, 17, 35, 53]
pair_ids = [(i, j) for i in prot_atom_ids for j in drug_atom_ids]

class BuildDataset(BaseDataset):
    def __init__(self, data_path, dataset, cut_dist, num_angle, save_file=True):
        self.data_path = data_path
        self.dataset = dataset
        self.cut_dist = cut_dist
        self.num_angle = num_angle
        self.save_file = save_file
        self.graph_prefix = f'{self.data_path}/{self.dataset}_{int(self.cut_dist)}_{self.num_angle}_pgl_graph'

        self.labels = []
        self.a2a_graphs = []
        self.b2a_graphs = []
        self.b2b_grpahs_list = []
        self.inter_feats_list = []
        self.bond_types_list = []
        self.type_count_list = []

        self.load_data()

    def __len__(self):
        """ Return the number of graphs. """
        return self.length

    def has_cache(self, idx):
        """ Check cache file."""
        graph_path = self.graph_prefix + f'_{idx}.pkl'
        return os.path.exists(graph_path)

    def save(self, idx, graphs, global_feat, label):
        """ Save the generated graphs. """
        #print('Saving graphs...')
        graph_path = self.graph_prefix + f'_{idx}.pkl'
        with open(graph_path, 'wb') as f:
            pickle.dump((graphs, global_feat, label), f, protocol=4)

    def build_graph(self, mol):
        num_atoms_d, coords, features, atoms, inter_feats = mol

        ##################################################
        # prepare distance matrix and interaction matrix #
        ##################################################
        dist_mat = distance.cdist(coords, coords, 'euclidean')
        np.fill_diagonal(dist_mat, np.inf)
        inter_feats = np.array([inter_feats])
        if inter_feats.sum() != 0:
            inter_feats = inter_feats / inter_feats.sum()

        ############################
        # build atom to atom graph #
        ############################
        num_atoms = len(coords)
        dist_graph_base = dist_mat.copy()
        dist_feat = dist_graph_base[dist_graph_base < self.cut_dist].reshape(-1, 1)
        dist_graph_base[dist_graph_base >= self.cut_dist] = 0.
        atom_graph = coo_matrix(dist_graph_base)
        a2a_edges = list(zip(atom_graph.row, atom_graph.col))
        a2a_graph = pgl.Graph(a2a_edges, num_nodes=num_atoms, node_feat={"feat": features},
                              edge_feat={"dist": dist_feat})

        ######################
        # prepare bond nodes #
        ######################
        indices = []
        bond_pair_atom_types = []
        for i in range(num_atoms):
            for j in range(num_atoms):
                a = dist_mat[i, j]
                if a < self.cut_dist:
                    at_i, at_j = atoms[i], atoms[j]
                    if i < num_atoms_d and j >= num_atoms_d and (at_j, at_i) in pair_ids:
                        bond_pair_atom_types += [pair_ids.index((at_j, at_i))]
                    elif i >= num_atoms_d and j < num_atoms_d and (at_i, at_j) in pair_ids:
                        bond_pair_atom_types += [pair_ids.index((at_i, at_j))]
                    else:
                        bond_pair_atom_types += [-1]
                    indices.append([i, j])

        ############################
        # build bond to atom graph #
        ############################
        num_bonds = len(indices)
        assignment_b2a = np.zeros((num_bonds, num_atoms), dtype=np.int64)  # Maybe need too much memory
        assignment_a2b = np.zeros((num_atoms, num_bonds), dtype=np.int64)  # Maybe need too much memory
        for i, idx in enumerate(indices):
            assignment_b2a[i, idx[1]] = 1
            assignment_a2b[idx[0], i] = 1

        b2a_graph = coo_matrix(assignment_b2a)
        b2a_edges = list(zip(b2a_graph.row, b2a_graph.col))
        b2a_graph = pgl.BiGraph(b2a_edges, src_num_nodes=num_bonds, dst_num_nodes=num_atoms)

        ############################
        # build bond to bond graph #
        ############################
        bond_graph_base = assignment_b2a @ assignment_a2b
        np.fill_diagonal(bond_graph_base, 0)  # eliminate self connections
        bond_graph_base[range(num_bonds), [indices.index([x[1], x[0]]) for x in indices]] = 0
        x, y = np.where(bond_graph_base > 0)
        num_edges = len(x)

        # calculate angle
        angle_feat = np.zeros_like(x, dtype=np.float32)
        for i in range(num_edges):
            body1 = indices[x[i]]
            body2 = indices[y[i]]
            a = dist_mat[body1[0], body1[1]]
            b = dist_mat[body2[0], body2[1]]
            c = dist_mat[body1[0], body2[1]]
            if a == 0 or b == 0:
                print(body1, body2)
                print('One distance is zero.')
                angle_feat[i] = 0.
                return None, None
                # exit(-1)
            else:
                angle_feat[i] = cos_formula(a, b, c)

        # angle domain divisions
        unit = 180.0 / self.num_angle
        angle_index = (np.rad2deg(angle_feat) / unit).astype('int64')
        angle_index = np.clip(angle_index, 0, self.num_angle - 1)

        # multiple bond-to-bond graphs based on angle domains
        b2b_edges_list = [[] for _ in range(self.num_angle)]
        b2b_angle_list = [[] for _ in range(self.num_angle)]
        for i, (ind, radian) in enumerate(zip(angle_index, angle_feat)):
            b2b_edges_list[ind].append((x[i], y[i]))
            b2b_angle_list[ind].append(radian)

        # b2b_graph_list = [[] for _ in range(self.num_angle)]
        b2b_graph_list = []
        for ind in range(self.num_angle):
            b2b_graph = pgl.Graph(b2b_edges_list[ind], num_nodes=num_bonds, edge_feat={"angle": b2b_angle_list[ind]})
            b2b_graph_list.append(b2b_graph)

        #########################################
        # build index for inter-molecular bonds #
        #########################################
        bond_types = bond_pair_atom_types
        type_count = [0 for _ in range(len(pair_ids))]
        for type_i in bond_types:
            if type_i != -1:
                type_count[type_i] += 1

        bond_types = np.array(bond_types)
        type_count = np.array(type_count)

        graphs = a2a_graph, b2a_graph, b2b_graph_list
        global_feat = inter_feats, bond_types, type_count
        return graphs, global_feat

    def load_data(self):
        """ Generate complex interaction graphs. """
        print('Processing raw protein-ligand complex data...')
        file_name = os.path.join(self.data_path, "{0}.pkl".format(self.dataset))
        with open(file_name, 'rb') as f:
            data_mols, data_Y = pickle.load(f)

        idx = 0
        for mol, y in tqdm(zip(data_mols, data_Y)):
            if self.has_cache(idx):
                idx += 1
                continue
            graphs, global_feat = self.build_graph(mol)
            if graphs is None:
                continue
            self.save(idx, graphs, global_feat, y)
            idx += 1
        self.length = idx


class ComplexDataset(BaseDataset):
    def __init__(self, data_path, dataset, cut_dist, num_angle, start, end, set_type, indices=None):
        self.data_path = data_path
        self.dataset = dataset
        self.cut_dist = cut_dist
        self.num_angle = num_angle
        self.graph_prefix = f'{self.data_path}/{self.dataset}_{int(self.cut_dist)}_{self.num_angle}_pgl_graph'

        self.labels = []
        self.a2a_graphs = []
        self.b2a_graphs = []
        self.b2b_grpahs_list = []
        self.inter_feats_list = []
        self.bond_types_list = []
        self.type_count_list = []

        self.load_data(start, end, set_type, indices)

    def __len__(self):
        """ Return the number of graphs. """
        return len(self.labels)

    def __getitem__(self, idx):
        """ Return graphs and label. """
        return self.a2a_graphs[idx], self.b2a_graphs[idx], self.b2b_grpahs_list[idx], \
               self.inter_feats_list[idx], self.bond_types_list[idx], self.type_count_list[idx], self.labels[idx]

    def load(self, idx):
        """ Load the generated graphs. """
        graph_path = self.graph_prefix + f'_{idx}.pkl'
        with open(graph_path, 'rb') as f:
            graphs, global_feat, score = pickle.load(f)
        a2a_graph, b2a_graph, b2b_graph = graphs
        inter_feats, bond_types, type_count = global_feat
        self.labels.append(score)
        self.a2a_graphs.append(a2a_graph)
        self.b2a_graphs.append(b2a_graph)
        self.b2b_grpahs_list.append(b2b_graph)
        self.inter_feats_list.append(inter_feats)
        self.bond_types_list.append(bond_types)
        self.type_count_list.append(type_count)

    def load_data(self, start, end, set_type, indices=None):
        print('Loading dataset...')
        if indices is None:
            for idx in tqdm(range(start, end + 1)):
                self.load(idx)
        else:
            for idx in tqdm(indices):
                self.load(idx)
        if set_type == 'training':
            df = pd.read_csv('../data/dataframe_37k_training.csv').query('rmsd < 1.5')
        elif set_type == 'validation':
            df = pd.read_csv('../data/dataframe_37k_validation.csv').query('rmsd < 1.5')
        else:
            df = pd.read_csv('../data/dataframe_37k_test.csv').query('rmsd < 1.5')
        #self.labels = (df['energy'] - df['e_docking']).values.reshape(-1, 1)
        self.labels = np.array(self.labels).reshape(-1, 1)

def collate_fn(batch):
    a2a_gs, b2a_gs, b2b_gs_l, feats, types, counts, labels = map(list, zip(*batch))

    a2a_g = pgl.Graph.batch(a2a_gs).tensor()
    b2a_g = pgl.BiGraph.batch(b2a_gs).tensor()
    b2b_gl = [pgl.Graph.batch([g[i] for g in b2b_gs_l]).tensor() for i in range(len(b2b_gs_l[0]))]
    feats = paddle.concat([paddle.to_tensor(f, dtype='float32') for f in feats])
    types = paddle.concat([paddle.to_tensor(t) for t in types])
    counts = paddle.stack([paddle.to_tensor(c) for c in counts], axis=1)
    labels = paddle.to_tensor(np.array(labels), dtype='float32')

    return a2a_g, b2a_g, b2b_gl, feats, types, counts, labels

if __name__ == "__main__":
    complex_data = ComplexDataset("./data/", "pdbbind2016_test", 5, 6)
    loader = Dataloader(complex_data,
                        batch_size=32,
                        shuffle=False,
                        num_workers=1,
                        collate_fn=collate_fn)
    cc = 0
    for batch in loader:
        a2a_g, b2a_g, b2b_gl, feats, types, counts, labels = batch
        print(labels)
        cc += 1
        if cc == 2:
            break
