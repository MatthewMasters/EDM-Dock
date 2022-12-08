import os
import re
from collections import defaultdict
from subprocess import call
from itertools import product

import torch
import parmed
import numpy as np
import mdtraj as md
from glob import glob
from rdkit import Chem
from rdkit import RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.rdPartialCharges import ComputeGasteigerCharges
from rdkit.Chem import Crippen
from rdkit.Chem import rdMolDescriptors

from edmdock.utils.esm import get_esm_embed, RESIDUE_TO_ESM_TOKEN
from edmdock.utils.utils import save_pickle, load_pickle
from edmdock.utils.chem import RESIDUES, ELEMENTS, HYBRIDIZATION_TYPES, PERIODIC_TABLE, BOND_TYPES, load_ligand
from edmdock.utils.data import PairData
from edmdock.utils.nn import random_rotate

RDLogger.DisableLog('rdApp.*')
FEATURE_SCALING = {
    'charges': [-0.18, 0.33],
    'weights': [14.15, 4.57],
    'radii': [1.63, 0.16],
    'sasa': [5.75, 1.02],
    'log_p': [-0.12, 0.49],
    'mr': [2.66, 1.65],
    'tpsa': [6.43, 8.84],
}


def one_hot(val, d):
    """numpy one-hot provided value and index dictionary"""
    v = np.zeros(max(d.values()) + 1)
    idx = d[val]
    if idx == -1:
        return v
    else:
        v[idx] = 1
        return v


def rbf(charges, N_kernels=16, gamma=32):
    mus = torch.linspace(-1, 1, N_kernels + 1)[:N_kernels]
    return torch.exp(-gamma * torch.square(charges[..., None] - mus))


def preprocess_features(feature_dict, feature_keys):
    features = []
    # provide list of keys so ensure presence and order
    for key in feature_keys:
        val = feature_dict[key]
        if key in FEATURE_SCALING.keys():
            mu, std = FEATURE_SCALING[key]
        else:
            mu, std = 0.0, 1.0
        val = (np.asarray(val) - mu) / std
        if len(val.shape) == 1:
            val = np.expand_dims(val, axis=1)
        features.append(val)
    features = np.hstack(features)
    features = np.clip(np.nan_to_num(features, 0.0), -1000.0, 1000.0)
    return features


def get_box(ligand_path):
    coords = np.array([l.split()[:3] for l in open(ligand_path, 'r').read().strip().split('\n') if l.count('.') == 3], dtype=float)
    center = np.mean(coords, axis=0)
    cx, cy, cz = center.tolist()
    corner_a = np.min(coords, axis=0)
    corner_b = np.max(coords, axis=0)
    size = corner_b - corner_a + 8
    size = np.clip(size, 22.5, np.inf)
    sx, sy, sz = size.tolist()
    return cx, cy, cz, sx, sy, sz


def generate_pocket(system_path):
    # check if step already complete
    output_path = os.path.join(system_path, 'pocket_idx.npz')
    if os.path.exists(output_path):
        return

    # load protein
    protein_path = os.path.join(system_path, 'protein.pdb')
    protein = parmed.load_file(protein_path)

    # keep C alpha atoms (not Calcium!)
    protein_ca = protein['@CA & !:CA']

    # extra step to avoid duplicate atoms
    one_atom_per_res = np.array([res.atoms[0].idx for res in protein_ca.residues])
    protein_ca = protein_ca[one_atom_per_res]

    # load box params
    box_path = os.path.join(system_path, 'box.csv')
    box = np.array(open(box_path, 'r').read().strip().split(','), dtype=float)
    center, size = box[:3], box[3:]

    # get lower and upper corner
    lower = center - size / 2.0
    upper = center + size / 2.0

    # find C alpha atoms inside box -> create pocket
    pos_ca = np.array([[c.x, c.y, c.z] for c in protein_ca.positions])
    inside_idx = np.argwhere(np.all((lower < pos_ca) & (pos_ca < upper), axis=1)).flatten()
    pocket_ca = protein_ca[inside_idx]

    # only one chain
    pocket_ca = pocket_ca[pocket_ca.residues[0].chain, :, :]
    protein_ca = protein_ca[pocket_ca.residues[0].chain, :, :]

    # again, get inside idx
    pos_ca = np.array([[c.x, c.y, c.z] for c in protein_ca.positions])
    inside_idx = np.argwhere(np.all((lower < pos_ca) & (pos_ca < upper), axis=1)).flatten()
    residue_idx = np.array([res.number for res in pocket_ca.residues])
    atom_idx = np.array([atom.number for atom in pocket_ca.atoms])

    # select full atomistic representation of pocket
    fa_inside_idx = np.array([a.idx for a in getattr(protein, 'atoms') if a.residue.idx in inside_idx])
    pocket = protein[fa_inside_idx]

    # save all structures and index of residues inside box
    pocket_ca.save(os.path.join(system_path, 'pocket_CA.pdb'), overwrite=True)
    protein_ca.save(os.path.join(system_path, 'protein_CA.pdb'), overwrite=True)
    pocket.save(os.path.join(system_path, 'pocket.pdb'), overwrite=True)

    np.savez(output_path, inside_idx=inside_idx, residue_idx=residue_idx, atom_idx=atom_idx)


def generate_pocket_multichain(system_path):
    # check if step already complete
    output_path = os.path.join(system_path, 'pocket_idx.npz')
    if os.path.exists(output_path):
        return

    # load protein
    protein_path = os.path.join(system_path, 'protein.pdb')
    protein = parmed.load_file(protein_path)

    # keep C alpha atoms (not Calcium!)
    protein_ca = protein['@CA & !:CA']

    # extra step to avoid duplicate atoms
    one_atom_per_res = np.array([res.atoms[0].idx for res in protein_ca.residues])
    protein_ca = protein_ca[one_atom_per_res]

    # load box params
    box_path = os.path.join(system_path, 'box.csv')
    box = np.array(open(box_path, 'r').read().strip().split(','), dtype=float)
    center, size = box[:3], box[3:]

    # get lower and upper corner
    lower = center - size / 2.0
    upper = center + size / 2.0

    # find C alpha atoms inside box -> create pocket
    pos_ca = np.array([[c.x, c.y, c.z] for c in protein_ca.positions])
    inside_idx = np.argwhere(np.all((lower < pos_ca) & (pos_ca < upper), axis=1)).flatten()
    pocket_ca = protein_ca[inside_idx]

    # save all structures and index of residues inside box
    pocket_ca.save(os.path.join(system_path, 'pocket_CA.pdb'), overwrite=True)
    protein_ca.save(os.path.join(system_path, 'protein_CA.pdb'), overwrite=True)

    np.savez(output_path, inside_idx=inside_idx)


def generate_simple(system_path):
    key = system_path.split('/')[-1]
    output_path = os.path.join(system_path, 'simple.pkl')
    if os.path.exists(output_path):
        return

    # load pocket
    pocket_path = os.path.join(system_path, 'pocket_CA.pdb')
    pocket = parmed.load_file(pocket_path)

    # generate pocket features
    pocket_n = len(pocket.atoms)
    pocket_types = np.array([RESIDUES[res.name] for res in pocket.residues])
    pocket_pos = np.array([[atom.xx, atom.xy, atom.xz] for atom in pocket.atoms])
    pocket_edge_index = np.array([[i, j] for i in range(pocket_n) for j in range(pocket_n) if i != j]).T

    # load ligand
    ligand_path = os.path.join(system_path, 'ligand.sdf')
    try:
        ligand = load_ligand(ligand_path)
        docked_pos = np.array(ligand.GetConformer(0).GetPositions(), dtype=float)
    except:
        return
    ligand_types = np.array([ELEMENTS[atom.GetSymbol()] for atom in ligand.GetAtoms()])
    ligand_n = len(docked_pos)

    # Generate conformer which is unbiased by the docked pose
    try:
        conf_id = AllChem.EmbedMolecule(ligand)
        ligand_pos = np.array(ligand.GetConformer(conf_id).GetPositions(), dtype=np.float32)
    except:
        # raise Exception(f'Could not generate RDKit conformer ({system_path})')
        ligand_pos = docked_pos + np.random.uniform(-20, 20, size=3)
        ligand_pos = random_rotate(torch.tensor(ligand_pos, dtype=torch.float32)).numpy()

    # Compute charges
    try:
        ComputeGasteigerCharges(ligand)
        charges = np.array([float(atom.GetProp('_GasteigerCharge')) for atom in ligand.GetAtoms()])
    except:
        charges = np.zeros(ligand_n)

    ligand_features = rbf(torch.tensor(charges), N_kernels=16, gamma=32)
    ligand_features = torch.clip(torch.nan_to_num(ligand_features, 0.0), -1000, 1000)
    # ligand_features = torch.unsqueeze(torch.clip(torch.nan_to_num(torch.tensor(charges), 0.0), -10, 10), dim=-1)

    # Calculate features and connectivity for ligand bonds
    ligand_edge_index = []
    ligand_edge_types = []
    for bond in ligand.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ligand_edge_index.extend([[i, j], [j, i]])
        type_ = BOND_TYPES[bond.GetBondType()]
        ligand_edge_types.extend([type_, type_])
    ligand_edge_index = np.array(ligand_edge_index, dtype=np.long).T
    ligand_edge_types = np.array(ligand_edge_types)

    inter_edge_index = np.array(list(product(range(ligand_n), range(pocket_n)))).T
    ligand_i, pocket_i = inter_edge_index
    dis_gt = np.linalg.norm(docked_pos[ligand_i] - pocket_pos[pocket_i], axis=1)

    data = PairData(
        key=key,
        ligand_features=ligand_features.float(),
        pocket_types=torch.tensor(pocket_types),
        ligand_types=torch.tensor(ligand_types),
        docked_pos=torch.tensor(docked_pos),
        ligand_pos=torch.tensor(ligand_pos),
        pocket_pos=torch.tensor(pocket_pos),
        ligand_edge_types=torch.tensor(ligand_edge_types, dtype=torch.long),
        ligand_edge_index=torch.tensor(ligand_edge_index, dtype=torch.long),
        pocket_edge_index=torch.tensor(pocket_edge_index, dtype=torch.long),
        inter_edge_index=torch.tensor(inter_edge_index, dtype=torch.long),
        dis_gt=torch.tensor(dis_gt),
    )
    save_pickle(output_path, data)


def generate_esm(system_path, embed_model, batch_converter, device):
    # check if step already complete
    output_path_a = os.path.join(system_path, 'esm_protein.npy')
    output_path_b = os.path.join(system_path, 'esm_pocket.npy')
    if os.path.exists(output_path_a) and os.path.exists(output_path_b):
        return

    # load protein
    protein_path = os.path.join(system_path, 'protein_CA.pdb')
    protein = parmed.load_file(protein_path)

    # filter sequences >1024 (limit of ESM model)
    if len(protein.residues) > 1024:
        print(f'protein sequence length cannot be >1024 to generate ESM embedding ({system_path})')
        return

    if os.path.exists(output_path_a):
        # load esm_protein if it exists
        esm_embed = np.load(output_path_a)
    else:
        # generate esm embedding for full protein
        esm_tokens = []
        pos_to_idx = {}
        for idx, res in enumerate(protein.residues):
            token = RESIDUE_TO_ESM_TOKEN[res.name]
            esm_tokens.append(token)
            pos_to_idx[res.number] = idx
        esm_tokens = torch.tensor(esm_tokens).to(device)
        esm_embed = get_esm_embed(esm_tokens, embed_model, batch_converter).squeeze().cpu().numpy()
        np.save(output_path_a, esm_embed)

    # load pocket residue index, use to select ESM embeddings for the pocket, and save
    index = np.load(os.path.join(system_path, 'pocket_idx.npz'))
    np.save(output_path_b, esm_embed[index['inside_idx']])


def generate_hb_data(pdb_path, pocket_mol):
    # run HBPLUS
    hb2_path = pdb_path.replace('.pdb', '.hb2')
    if not os.path.exists(hb2_path):
        cmd = f'/data/masters/tools/HBPLUS/hbplus/hbplus {os.path.basename(pdb_path)}'
        call(cmd, shell=True, cwd=os.path.dirname(pdb_path))

    # parse results and gather data in dict
    hb_data = defaultdict(int)
    lines = open(hb2_path, 'r').read().strip().split('\n')[8:]
    for line in lines:
        donor, _, acceptor, _, _, hb_type, *_ = line.split()
        try:
            donor_id = int(re.sub('\D', '', donor.strip('-').split('-')[0]))
            acceptor_id = int(re.sub('\D', '', donor.strip('-').split('-')[0]))
        except:
            continue
        hb_types = ['MM', 'MS', 'SM', 'SS']
        if hb_type not in hb_types:
            continue
        hb_type = hb_types.index(hb_type)
        v = np.zeros(4)
        v[hb_type] += 1
        hb_data[donor_id] += np.concatenate([np.zeros(4), v])
        hb_data[acceptor_id] += np.concatenate([v, np.zeros(4)])

    # if a residue has no HB data, assign a zeros vector
    for res in pocket_mol.residues:
        if type(hb_data[res.number]) == int:
            hb_data[res.number] = np.zeros(8)

    return hb_data


def generate_disulfide_data(protein_mol):
    disulfides = defaultdict(int)
    for res in protein_mol.residues:
        flag = 0
        if res.name == 'CYS':
            atoms = [atom.name for atom in res.atoms]
            if 'HG' not in atoms:
                flag = 1
        disulfides[res.number] = flag
    return disulfides


def generate_sasa_data(pdb_path):
    md_mol = md.load(pdb_path)
    keys, keep_idx, cas = [], [], []
    for atom in md_mol.topology.atoms:
        key = f'{atom.residue}-{atom.name}'
        if key not in keys:
            keys.append(key)
            keep_idx.append(atom.index)
            if atom.name == 'CA':
                cas.append(len(keep_idx) - 1)
    md_mol = md_mol.atom_slice(atom_indices=keep_idx)
    try:
        sasa = md.shrake_rupley(md_mol)
    except:
        raise Exception('Could not generate SASA features')
    sasa = sasa[0, np.array(cas)]
    return sasa


def generate_protein_features(system_path, gen_sasa=False, gen_hb=False):
    output_path = os.path.join(system_path, 'protein_data.pkl')
    if os.path.exists(output_path):
        return
    protein_path = os.path.join(system_path, 'protein.pdb')
    pocket_path = os.path.join(system_path, 'pocket_CA.pdb')

    # Load pocket and protein
    pocket_mol = parmed.load_file(pocket_path)
    index = np.load(os.path.join(system_path, 'pocket_idx.npz'))
    protein_mol = parmed.load_file(protein_path)

    pocket_coords = np.array([[c.x, c.y, c.z] for c in getattr(pocket_mol, 'positions')])

    protein_length = len(protein_mol.residues)
    disulfides = generate_disulfide_data(protein_mol)
    features = defaultdict(list)
    features['residues'] = [one_hot(res.name, RESIDUES) for res in pocket_mol.residues]
    features['position'] = [idx / protein_length for idx in index['residue_idx']]
    features['disulfides'] = [disulfides[idx] for idx in index['residue_idx']]
    if gen_sasa:
        sasa = generate_sasa_data(protein_path)
        features['sasa'] = [sasa[idx] for idx in index['atom_idx']]
    if gen_hb:
        hb_data = generate_hb_data(protein_path, pocket_mol)
        features['hbond'] = [hb_data[idx] for idx in index['residue_idx']]

    idx = list(range(len(pocket_coords)))
    bond_index = np.array(list(product(idx, idx))).T

    data = {
        'pocket_mol': pocket_mol,
        'protein_mol': protein_mol,
        'coords': pocket_coords,
        'features': features,
        'bond_index': bond_index
    }
    save_pickle(output_path, data)


def generate_ligand_features(system_path):
    output_path = os.path.join(system_path, 'ligand_data.pkl')
    if os.path.exists(output_path):
        return

    # Find ligand input file
    try:
        ligand_path = glob(os.path.join(system_path, 'ligand.*'))[0]
    except:
        raise FileNotFoundError(f'Could not find ligand file ({system_path})')

    mol = load_ligand(ligand_path)

    # compute partial charge, LogP, ASA, and TPSA contributions
    ComputeGasteigerCharges(mol)
    crippen_contribs = Crippen._GetAtomContribs(mol)
    asa_contribs = rdMolDescriptors._CalcLabuteASAContribs(mol)[0]
    tpsa_contribs = rdMolDescriptors._CalcTPSAContribs(mol)

    # Get smiles for ligand
    smiles = Chem.MolToSmiles(mol)

    # Get coordinates
    coords = np.array(mol.GetConformer(0).GetPositions(), dtype=np.float32)

    # Generate conformer which is unbiased by the docked pose
    try:
        conf_id = AllChem.EmbedMolecule(mol)
        rdkit_coords = np.array(mol.GetConformer(conf_id).GetPositions(), dtype=np.float32)
    except:
        # raise Exception(f'Could not generate RDKit conformer ({system_path})')
        rdkit_coords = coords + np.random.uniform(-20, 20, size=(3,))
        rdkit_coords = random_rotate(torch.tensor(rdkit_coords, dtype=torch.float32)).numpy()

    # Calculate atom features for ligand atoms
    features = defaultdict(list)
    for ai, atom in enumerate(mol.GetAtoms()):
        features['charges'].append(float(atom.GetProp('_GasteigerCharge')))
        features['aromaticity'].append(int(atom.GetIsAromatic()))
        features['element'].append(one_hot(atom.GetSymbol(), ELEMENTS))
        features['hybridization'].append(one_hot(atom.GetHybridization(), HYBRIDIZATION_TYPES))
        features['weights'].append(atom.GetMass())
        features['radii'].append(PERIODIC_TABLE.van_der_waals_radius[atom.GetAtomicNum()])
        features['sasa'].append(asa_contribs[ai])
        features['log_p'].append(crippen_contribs[ai][0])
        features['mr'].append(crippen_contribs[ai][1])
        features['tpsa'].append(tpsa_contribs[ai])

    for key, val in features.items():
        features[key] = np.nan_to_num(np.vstack(val))

    # Calculate features and connectivity for ligand bonds
    bond_index = []
    bond_types = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_index.extend([[i, j], [j, i]])
        type_ = BOND_TYPES[bond.GetBondType()]
        bond_types.extend([type_, type_])
    bond_index = np.array(bond_index, dtype=np.long).T
    bond_types = np.array(bond_types)

    data = {
        'mol': mol,
        'smiles': smiles,
        'docked_coords': coords,
        'ligand_coords': rdkit_coords,
        'features': features,
        'bond_index': bond_index,
        'bond_types': bond_types
    }
    save_pickle(output_path, data)


def generate_batch_data(system_path):
    output_path = os.path.join(system_path, 'data_esm.pkl')
    # if os.path.exists(output_path):
    #     return

    key = os.path.basename(system_path)
    try:
        ligand_data = load_pickle(os.path.join(system_path, 'ligand_data.pkl'))
        protein_data = load_pickle(os.path.join(system_path, 'protein_data.pkl'))
        pocket_esm = np.load(os.path.join(system_path, 'esm_pocket.npy'))
    except:
        # not all required files are present, skipping
        return

    ligand_coords = ligand_data['ligand_coords']
    docked_coords = ligand_data['docked_coords']
    pocket_coords = protein_data['coords']

    n_ligand = len(docked_coords)
    n_pocket = len(pocket_coords)

    inter_edge_index = np.array(list(product(range(n_ligand), range(n_pocket)))).T
    ligand_i, pocket_i = inter_edge_index
    dis_gt = np.linalg.norm(docked_coords[ligand_i] - pocket_coords[pocket_i], axis=1)

    ligand_keys = ['charges', 'aromaticity', 'element', 'hybridization', 'weights', 'radii', 'sasa', 'log_p', 'mr',
                   'tpsa']
    ligand_features = preprocess_features(ligand_data['features'], ligand_keys)
    print(ligand_features.shape)
    # print(ligand_data['features'])
    pocket_keys = ['residues', 'position', 'disulfides']
    pocket_features = preprocess_features(protein_data['features'], pocket_keys)
    pocket_features = np.hstack([pocket_features, pocket_esm])

    pocket_edge_index = np.array([[i, j] for i in range(n_pocket) for j in range(n_pocket) if i != j]).T

    data = PairData(
        key=key,
        ligand_features=torch.tensor(ligand_features),
        bond_types=torch.tensor(ligand_data['bond_types']),
        pocket_features=torch.tensor(pocket_features),
        docked_pos=torch.tensor(docked_coords),
        ligand_pos=torch.tensor(ligand_coords),
        pocket_pos=torch.tensor(pocket_coords),
        ligand_edge_index=torch.tensor(ligand_data['bond_index'], dtype=torch.long),
        pocket_edge_index=torch.tensor(pocket_edge_index, dtype=torch.long),
        inter_edge_index=torch.tensor(inter_edge_index, dtype=torch.long),
        dis_gt=torch.tensor(dis_gt),
    )
    save_pickle(output_path, data)
