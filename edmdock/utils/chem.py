import os

import pandas as pd
from rdkit import Chem
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem.rdchem import HybridizationType as HT

ELEMENTS = {
    'H':  0,
    'C':  1,
    'O':  2,
    'N':  3,
    'S':  4,
    'P':  5,
    'B':  6,
    'Cl': 7,
    'F':  7,
    'I':  7,
    'Br': 7,
    'Se': 8,
    'As': 8,
    'Ir': 8,
    'Sb': 8,
    'Ru': 8,
    'Fe': 8,
    'Si': 8,
    'Zn': 8,
    'Pt': 8,
    'V':  8,
    'Co': 8,
    'Cu': 8,
    'Te': 8,
    'Rh': 8,
    'Re': 8,
    'Mg': 8,
    'Ca': 8,
    'Al': 8,
    'Ta': 8,
    'Mo': 8,
    'Hg': 8,
    'W':  8,
    'Y':  8,
    'Hf': 8,
    'Sn': 8,
    'Pd': 8,
    'Cr': 8,
    'Os': 8,
    'Pa': 8,
    'Ga': 8,
    'Mn': 8,
    'Ni': 8,
    'Zr': 8
}
RESIDUES = {
    'ALA': 0,
    'GLY': 1,
    'ARG': 2,
    'ASN': 3,
    'ASP': 4,
    'ASH': 4,
    'CSD': 4,  # 3-SULFINOALANINE
    'CYS': 5,
    'CYM': 5,
    'CYX': 5,
    'CAS': 5,  # S-DIMETHYLARSENIC-CYSTEINE
    'CAF': 5,  # S-DIMETHYLARSINOYL-CYSTEINE
    'CSO': 5,  # S-HYDROXYCYSTEINE
    'GLN': 6,
    'GLU': 7,
    'GLH': 7,
    'PCA': 7,  # PYROGLUTAMIC ACID
    'HIE': 8,
    'HIS': 8,
    'HID': 8,
    'HIP': 8,
    'ILE': 9,
    'LEU': 10,
    'LYS': 11,
    'KCX': 11,  # LYSINE NZ-CARBOXYLIC ACID
    'MLY': 11,  # N-DIMETHYL-LYSINE
    'MET': 12,
    'MSE': 12,  # SELENOMETHIONINE
    'PHE': 13,
    'PRO': 14,
    'SER': 15,
    'SEP': 15,  # PHOSPHOSERINE
    'THR': 16,
    'TPO': 16,  # PHOSPHOTHREONINE
    'TRP': 17,
    'TYR': 18,
    'PTR': 18,  # PHOSPHOTYROSINE
    'VAL': 19,
    'LLP': 20,  # NON STANDARD
    'UNK': 20,
    'HIZ': 20,
    'GLZ': 20,
    'SEM': 20,
    'ASQ': 20,
    'TYM': 20,
    'LEV': 20,
    'MEU': 20,
    'ASM': 20,
    'DIC': 20,
    'ALB': 20,
    'DID': 20,
    'GLV': 20,
    'GLO': 20,
    'CYT': 20,
    'GLM': 20,
    'HIY': 20,
    'STA': 20
}
THREE_TO_ONE_LETTER_MAP = {
    'ALA': 'A',
    'GLY': 'G',
    'ARG': 'R',
    'ASN': 'N',
    'ASP': 'D',
    'ASH': 'D',
    'CSD': 'D',  # 3-SULFINOALANINE
    'CYS': 'C',
    'CYM': 'C',
    'CYX': 'C',
    'CAS': 'C',  # S-DIMETHYLARSENIC-CYSTEINE
    'CAF': 'C',  # S-DIMETHYLARSINOYL-CYSTEINE
    'CSO': 'C',  # S-HYDROXYCYSTEINE
    'GLN': 'Q',
    'GLU': 'E',
    'GLH': 'E',
    'PCA': 'E',  # PYROGLUTAMIC ACID
    'HIE': 'H',
    'HIS': 'H',
    'HID': 'H',
    'ILE': 'I',
    'LEU': 'L',
    'LYS': 'K',
    'KCX': 'K',  # LYSINE NZ-CARBOXYLIC ACID
    'MLY': 'K',  # N-DIMETHYL-LYSINE
    'MET': 'M',
    'MSE': 'M',  # SELENOMETHIONINE
    'PHE': 'F',
    'PRO': 'P',
    'SER': 'S',
    'SEP': 'S',  # PHOSPHOSERINE
    'THR': 'T',
    'TPO': 'T',  # PHOSPHOTHREONINE
    'TRP': 'W',
    'TYR': 'Y',
    'PTR': 'Y',  # PHOSPHOTYROSINE
    'VAL': 'V',
    'LLP': '_',  # NON STANDARD
    'ACE': '_'
}
AA_MAP = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19
}
ONE_TO_THREE_LETTER_MAP = {
    "R": "ARG",
    "H": "HIS",
    "K": "LYS",
    "D": "ASP",
    "E": "GLU",
    "S": "SER",
    "T": "THR",
    "N": "ASN",
    "Q": "GLN",
    "C": "CYS",
    "G": "GLY",
    "P": "PRO",
    "A": "ALA",
    "V": "VAL",
    "I": "ILE",
    "L": "LEU",
    "M": "MET",
    "F": "PHE",
    "Y": "TYR",
    "W": "TRP"
}
HYBRIDIZATION_TYPES = {
    HT.S: 0,
    HT.SP: 1,
    HT.SP2: 2,
    HT.SP3: 3,
    HT.SP3D: 4,
    HT.SP3D2: 5,
    HT.UNSPECIFIED: -1
}
BOND_TYPES = {
    BT.SINGLE: 0,
    BT.DOUBLE: 1,
    BT.TRIPLE: 2,
    BT.AROMATIC: 3,
    BT.UNSPECIFIED: 4
}
PERIODIC_TABLE = pd.read_csv(os.path.join(os.path.dirname(os.path.realpath(__file__)), 'periodic_table.csv'), index_col=0)


def load_ligand(ligand_path):
    # Load ligand molecule into rdkit
    ext = ligand_path.split('.')[-1]
    if ext == 'sdf':
        mol = next(Chem.SDMolSupplier(ligand_path))
        # TODO: tmp
        if mol is None:
            mol = Chem.MolFromMol2File(ligand_path.replace('.sdf', '.mol2'))
        if mol is None:
            mol = Chem.MolFromPDBFile(ligand_path.replace('.sdf', '.pdb'))
    elif ext == 'mol2':
        mol = Chem.MolFromMol2File(ligand_path)
        # TODO: tmp
        if mol is None:
            mol = next(Chem.SDMolSupplier(ligand_path.replace('.mol2', '.sdf')))
        if mol is None:
            mol = Chem.MolFromPDBFile(ligand_path.replace('.mol2', '.pdb'))
    elif ext == 'pdb':
        mol = Chem.MolFromPDBFile(ligand_path)
    else:
        raise Exception(f'Ligand filetype not supported ({ligand_path})')

    if mol is None:
        raise Exception(f'Ligand could not be loaded by RDkit ({ligand_path})')

    return mol


def write_xyz(filepath, coords):
    with open(filepath, 'w') as outfile:
        outfile.write(f'{len(coords)}\n')
        outfile.write('\n')
        for xyz in coords:
            outfile.write(f'O\t{xyz[0]}\t{xyz[1]}\t{xyz[2]}\n')

