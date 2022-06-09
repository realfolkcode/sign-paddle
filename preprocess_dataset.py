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
import pickle
import re
from tqdm import tqdm
import pandas as pd
import numpy as np
import argparse
from openbabel import openbabel
from openbabel import pybel
from featurizer import Featurizer
from scipy.spatial import distance_matrix
import time
import multiprocessing as mp
openbabel.obErrorLog.SetOutputLevel(0)

from pandarallel import pandarallel

def file_extension (filename):
    return os.path.splitext(filename)[1][1:]

def pocket_atom_num_from_mol2(pocket_name):
    n = 0
    with open(pocket_name) as f:
        for line in f:
            if '<TRIPOS>ATOM' in line:
                break
        for line in f:
            cont = line.split()
            if '<TRIPOS>BOND' in line or cont[7] == 'HOH':
                break
            n += int(cont[5].split(".")[0] != 'H')
    return n

def pocket_atom_num_from_pdb (pocket_name):
    n = 0
    with open(pocket_name) as f:
        for line in f:
            if 'REMARK' in line:
                break
        for line in f:
            cont = line.split()
            # break
            if cont[0] == 'CONECT':
                break
            n += int(cont[-1] != 'H' and cont[0] == 'ATOM')
    return n

def pocket_atom_num (pocket_name): 
    ext = file_extension(pocket_name)
    if ext == 'mol2':
        return pocket_atom_num_from_mol2(pocket_name)
    elif ext == 'pdb':
        return pocket_atom_num_from_pdb(pocket_name)
    else:
        raise ValueError("Incorrect file extension for pocket. Expected mol2 or pdb")

def read_molecule(molecule_name):
    '''Use pybel to load molecule. Format will be determined by extension'''
    fmt = file_extension(molecule_name)
    if fmt in pybel.informats:
        return pybel.readfile(fmt, molecule_name)
    else:
        raise ValueError("Unsupported molecule file format " + fmt + " while loading " + molecule_name)
    
## function -- feature
#def gen_feature(ligand_name, pocket_name, featurizer):
def gen_feature(ligand_name, pocket_name):

    charge_idx = featurizer.FEATURE_NAMES.index('partialcharge')
    
    try:
        ligand = next(read_molecule(ligand_name))
    except StopIteration:
        print("Problems with loading ligand from", ligand_name)
        raise ValueError("Ligand loading error")

    ligand_coords, ligand_features = featurizer.get_features(ligand, molcode=1)

    try:
        pocket = next(read_molecule(pocket_name))
    except StopIteration:
        print("Problems with loading pocket from ", pocket_name)
        raise ValueError("Pocket loading error")

    pocket_coords, pocket_features = featurizer.get_features(pocket, molcode=-1)
    node_num = pocket_atom_num(pocket_name)
    #node_num = pocket_atom_num_from_mol2(pocket_name)
    pocket_coords = pocket_coords[:node_num]
    pocket_features = pocket_features[:node_num]

    try:
        assert (ligand_features[:, charge_idx] != 0).any()
        assert (pocket_features[:, charge_idx] != 0).any()
        assert (ligand_features[:, :9].sum(1) != 0).all()
    except:
        print([ligand_name, pocket_name])
        return {'error':1}

    # try:
    #     assert (ligand_features[:, charge_idx] != 0).any()
    # except:
    #     print("Zero charges on ligand ", ligand_name, pocket_name)
    #     raise ValueError("All charges on a ligand are zero")

    # try:
    #     assert (ligand_features[:, :9].sum(1) != 0).all()
    # except:
    #     print("Ambiguous atom types on ligand ", ligand_name, pocket_name)
    #     raise ValueError("Ambiguous atom types")

    # try:
    #     assert (pocket_features[:, charge_idx] != 0).any()
    # except:
    #     print("Zero charges on pocket ", ligand_name, pocket_name)
    #     raise ValueError("All charges for a pocket are zero")

    lig_atoms, pock_atoms = [], []
    for i, atom in enumerate(ligand):
        if atom.atomicnum > 1:
            lig_atoms.append(atom.atomicnum)
    for i, atom in enumerate(pocket):
        if atom.atomicnum > 1:
            pock_atoms.append(atom.atomicnum)
    for x in pock_atoms[node_num:]:
        assert x == 8
    pock_atoms = pock_atoms[:node_num]
    assert len(lig_atoms)==len(ligand_features) and len(pock_atoms)==len(pocket_features)
    
    ligand_edges = gen_pocket_graph(ligand)
    pocket_edges = gen_pocket_graph(pocket)
    return {'lig_co': ligand_coords, 'lig_fea': ligand_features, 'lig_atoms': lig_atoms, 'lig_eg': ligand_edges, 'pock_co': pocket_coords, 'pock_fea': pocket_features, 'pock_atoms': pock_atoms, 'pock_eg': pocket_edges}

## function -- pocket graph
def gen_pocket_graph(pocket):
    edge_l = []
    idx_map = [-1]*(len(pocket.atoms)+1)
    idx_new = 0
    for atom in pocket:
        edges = []
        a1_sym = atom.atomicnum
        a1 = atom.idx
        if a1_sym == 1:
            continue
        idx_map[a1] = idx_new
        idx_new += 1
        for natom in openbabel.OBAtomAtomIter(atom.OBAtom):
            if natom.GetAtomicNum() == 1:
                continue
            a2 = natom.GetIdx()
            bond = openbabel.OBAtom.GetBond(natom,atom.OBAtom)
            bond_t = bond.GetBondOrder()
            edges.append((a1,a2,bond_t))
        edge_l += edges
    edge_l_new = []
    for a1,a2,t in edge_l:
        a1_, a2_ = idx_map[a1], idx_map[a2]
        assert((a1_!=-1)&(a2_!=-1))
        edge_l_new.append((a1_,a2_,t))
    return edge_l_new

def dist_filter(dist_matrix, theta): 
    pos = np.where(dist_matrix<=theta)
    ligand_list, pocket_list = pos
    return ligand_list, pocket_list

def pairwise_atomic_type(ligand_name, protein_name, atom_types, atom_types_, keys): 
    ligand = next(read_molecule(ligand_name))
    pocket = next(read_molecule(protein_name))

    coords_lig = np.vstack([atom.coords for atom in ligand])
    coords_poc = np.vstack([atom.coords for atom in pocket])
    atom_map_lig = [atom.atomicnum for atom in ligand]
    atom_map_poc = [atom.atomicnum for atom in pocket]
    dm = distance_matrix(coords_lig, coords_poc)
    # print(coords_lig.shape, coords_poc.shape, dm.shape)
    ligs, pocks = dist_filter(dm, 12)
    # print(len(ligs),len(pocks))
    
    fea_dict = {k: 0 for k in keys}
    for x, y in zip(ligs, pocks):
        x, y = atom_map_lig[x], atom_map_poc[y]
        if x not in atom_types or y not in atom_types_: continue
        fea_dict[(y, x)] += 1

    return list(fea_dict.values())

def get_lig_atom_types(feat):
    pos = np.where(feat[:,:9]>0)
    src_list, dst_list = pos
    return dst_list

def get_pock_atom_types(feat):
    pos = np.where(feat[:,18:27]>0)
    src_list, dst_list = pos
    return dst_list

def cons_spatial_gragh(dist_matrix, theta=5):
    pos = np.where((dist_matrix<=theta)&(dist_matrix!=0))
    src_list, dst_list = pos
    dist_list = dist_matrix[pos]
    edges = [(x,y) for x,y in zip(src_list, dst_list)]
    return edges, dist_list

def cons_mol_graph(edges, feas):
    size = feas.shape[0]
    edges = [(x,y) for x,y,t in edges]
    return size, feas, edges

def pocket_subgraph(node_map, edge_list, pock_dist):
    edge_l = []
    dist_l = []
    node_l = set()
    for coord, dist in zip(edge_list, np.concatenate([pock_dist, pock_dist])):
        x,y = coord
        if x in node_map and y in node_map:
            x, y = node_map[x], node_map[y]
            edge_l.append((x,y))
            dist_l.append(dist)
            node_l.add(x)
            node_l.add(y)
    dist_l = np.array(dist_l)
    return edge_l, dist_l

def edge_ligand_pocket(dist_matrix, lig_size, theta=4, keep_pock=False, reset_idx=True):
    
    pos = np.where(dist_matrix<=theta)
    ligand_list, pocket_list = pos
    if keep_pock:
        node_list = range(dist_matrix.shape[1])
    else:
        node_list = sorted(list(set(pocket_list)))
    node_map = {node_list[i]:i+lig_size for i in range(len(node_list))}
    
    dist_list = dist_matrix[pos]
    if reset_idx:
        edge_list = [(x,node_map[y]) for x,y in zip(ligand_list, pocket_list)]
    else:
        edge_list = [(x,y) for x,y in zip(ligand_list, pocket_list)]
    
    edge_list += [(y,x) for x,y in edge_list]
    dist_list = np.concatenate([dist_list, dist_list])
    
    return dist_list, edge_list, node_map

def add_identity_fea(lig_fea, pock_fea, comb=1):
    if comb == 1:
        lig_fea = np.hstack([lig_fea, [[1]]*len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[-1]]*len(pock_fea)])
    elif comb == 2:
        lig_fea = np.hstack([lig_fea, [[1,0]]*len(lig_fea)])
        pock_fea = np.hstack([pock_fea, [[0,1]]*len(pock_fea)])
    else:
        lig_fea = np.hstack([lig_fea, [[0]*lig_fea.shape[1]]*len(lig_fea)])
        if len(pock_fea) > 0:
            pock_fea = np.hstack([[[0]*pock_fea.shape[1]]*len(pock_fea), pock_fea])
    
    return lig_fea, pock_fea

def cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=2, theta=5, keep_pock=False, pocket_spatial=True):
    lig_fea, lig_coord, lig_atoms_raw, lig_edge = ligand
    pock_fea, pock_coord, pock_atoms_raw, pock_edge = pocket
    
    # inter-relation between ligand and pocket
    lig_size = lig_fea.shape[0]
    dm = distance_matrix(lig_coord, pock_coord)
    lig_pock_dist, lig_pock_edge, node_map = edge_ligand_pocket(dm, lig_size, theta=theta, keep_pock=keep_pock)

    # construct ligand graph & pocket graph
    lig_size, lig_fea, lig_edge = cons_mol_graph(lig_edge, lig_fea)
    pock_size, pock_fea, pock_edge = cons_mol_graph(pock_edge, pock_fea)
    
    # construct spatial context graph based on distance
    dm = distance_matrix(lig_coord, lig_coord)
    edges, lig_dist = cons_spatial_gragh(dm, theta=theta)
    if pocket_spatial:
        dm_pock = distance_matrix(pock_coord, pock_coord)
        edges_pock, pock_dist = cons_spatial_gragh(dm_pock, theta=theta)
    lig_edge = edges
    pock_edge = edges_pock
    
    # map new pocket graph
    pock_size = len(node_map)
    pock_fea = pock_fea[sorted(node_map.keys())]
    pock_edge, pock_dist = pocket_subgraph(node_map, pock_edge, pock_dist)
    pock_coord_ = pock_coord[sorted(node_map.keys())]
    
    # construct ligand-pocket graph
    size = lig_size + pock_size
    lig_fea, pock_fea = add_identity_fea(lig_fea, pock_fea, comb=add_fea)

    feas = np.vstack([lig_fea, pock_fea]) if len(pock_fea) > 0 else lig_fea
    edges = lig_edge + lig_pock_edge + pock_edge
    lig_atoms = get_lig_atom_types(feas)
    pock_atoms = get_pock_atom_types(feas)
    assert len(lig_atoms) ==  lig_size and len(pock_atoms) == pock_size
    
    atoms = np.concatenate([lig_atoms, pock_atoms]) if len(pock_fea) > 0 else lig_atoms
    
    lig_atoms_raw = np.array(lig_atoms_raw)
    pock_atoms_raw = np.array(pock_atoms_raw)
    pock_atoms_raw = pock_atoms_raw[sorted(node_map.keys())]
    atoms_raw = np.concatenate([lig_atoms_raw, pock_atoms_raw]) if len(pock_atoms_raw) > 0 else lig_atoms_raw
     
    coords = np.vstack([lig_coord, pock_coord_]) if len(pock_fea) > 0 else lig_coord
    if len(pock_fea) > 0:
        assert size==max(node_map.values())+1
    assert feas.shape[0]==coords.shape[0]
    return lig_size, coords, feas, atoms

def random_split(dataset_size, split_ratio=0.9, seed=0, shuffle=True):
    """random splitter"""
    np.random.seed(seed)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    split = int(split_ratio * dataset_size)
    train_idx, valid_idx = indices[:split], indices[split:]
    return train_idx, valid_idx

def add_y(df):
    df['value'] = 0.5**(((df['rmsd'] - 1.5) * 4)**2 / 4)
    df.loc[df['rmsd'] < 1.5, ['value']] = 1
    df.loc[df['rmsd'] > 2, ['value']] = 0.5**((df['rmsd']**2) / 4)
    df['value'] = df.value.array*df.energy.array
    return df

def construct_data(dataframe, cutoff):
    """Constructs datasets from processed dataframe"""

    data_dict = dict(zip(dataframe.name, dataframe.result))

    set_data, set_pk = [], []

    for k, v in tqdm(data_dict.items()):
        ligand = (v['lig_fea'], v['lig_co'], v['lig_atoms'], v['lig_eg'])
        pocket = (v['pock_fea'], v['pock_co'], v['pock_atoms'], v['pock_eg'])
        graph = cons_lig_pock_graph_with_spatial_context(ligand, pocket, add_fea=3, theta=cutoff, keep_pock=False, pocket_spatial=True)
        cofeat, pk = v['type_pair'], v['pk']
        graph = list(graph) + [cofeat]
        set_data.append(graph)
        set_pk.append(pk)
    
    return (set_data, set_pk)

def process_dataset(dataset_source, output_path, cutoff, dataset_name):
    """Read dataset from dataset_name. Save processed dataset to output_path 
    Paths in dataset are relative to the file with dataset """

    pandarallel.initialize(progress_bar=True, use_memory_fs=False)
    
    if dataset_source.endswith('.tsv'):
       df = pd.read_csv(dataset_source, sep='\t')
    elif dataset_source.endswith('.csv'):
       df = pd.read_csv(dataset_source)

    # add label
    if 'value' not in df.columns:
        if 'energy' in df.columns and 'rmsd' in df.columns:
            df = add_y(df)
        else:
            raise ValueError('Needs label')

    # Updating paths to relative to the dataset_name
    rootdir = os.path.dirname(dataset_source)
    df['ligand'] = df['ligand'].parallel_apply(lambda row: os.path.join(rootdir, row))
    df['pocket'] = df['pocket'].parallel_apply(lambda row: os.path.join(rootdir, row))
    df['protein'] = df['protein'].parallel_apply(lambda row: os.path.join(rootdir, row))

    # atomic sets for long-range interactions
    atom_types    = [6,7,8,9,15,16,17,35,53]
    atom_types_   = [6,7,8,16]

    # atomic feature generation
    print("Generating features for", len(df), "protein-ligand pairs")
    tic = time.perf_counter()
    df['result'] = df.parallel_apply(lambda x: gen_feature(x.ligand, x.pocket), axis = 1)    
    toc = time.perf_counter()
    print(f"Generated atomic features in {toc - tic:0.4f} seconds")
    print(df['result'].isna().sum(), "problematic complexes are excluded")
    df.dropna(subset=['result'], inplace=True)

    df.to_csv(dataset_name + '_atomic_features.csv')
    
    # interaction features
    print("Calculating interaction features:")
    keys = [(i,j) for i in atom_types_ for j in atom_types]
    tic = time.perf_counter()
    df['result'] = df.parallel_apply(lambda x: {**x.result,
        **{'type_pair':pairwise_atomic_type(x.ligand,x.protein, atom_types, atom_types_, keys)},
        **{'pk': x.value}}, axis=1)
    toc = time.perf_counter()
    print(f"Calculated interaction features in {toc - tic:0.4f} seconds")

    df.to_csv(dataset_name + '_interaction_features.csv')

    # save datasets to files
    all_dataset = construct_data(df, cutoff)
    with open(os.path.join(output_path, dataset_name + '.pkl'), 'wb') as f:
        pickle.dump(all_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_file', type=str)
    parser.add_argument('--output_path', type=str, default='./data/')
    parser.add_argument('--cutoff', type=float, default=5.)
    parser.add_argument('--dataset_name', type=str, default='dataset')
    args = parser.parse_args()
    #process_dataset(args.data_path_core, args.data_path_refined, args.dataset_name, args.output_path, args.cutoff)
    featurizer = Featurizer(save_molecule_codes=False)
    process_dataset(args.dataset_file, args.output_path, args.cutoff, args.dataset_name)
