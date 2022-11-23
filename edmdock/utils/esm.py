import torch

from .chem import AA_MAP, ONE_TO_THREE_LETTER_MAP


RESIDUE_TO_ESM_TOKEN = {
    'ALA': 0,
    'GLY': 5,
    'ARG': 14,
    'ASN': 11,
    'ASP': 2,
    'ASH': 2,
    'CSD': 2,  # 3-SULFINOALANINE
    'CYS': 1,
    'CYM': 1,
    'CYX': 1,
    'CAS': 1,  # S-DIMETHYLARSENIC-CYSTEINE
    'CAF': 1,  # S-DIMETHYLARSINOYL-CYSTEINE
    'CSO': 1,  # S-HYDROXYCYSTEINE
    'GLN': 13,
    'GLU': 3,
    'GLH': 3,
    'PCA': 3,  # PYROGLUTAMIC ACID
    'HIE': 6,
    'HIS': 6,
    'HID': 6,
    'ILE': 7,
    'LEU': 9,
    'LYS': 8,
    'KCX': 8,  # LYSINE NZ-CARBOXYLIC ACID
    'MLY': 8,  # N-DIMETHYL-LYSINE
    'MET': 10,
    'MSE': 10,  # SELENOMETHIONINE
    'PHE': 4,
    'PRO': 12,
    'SER': 15,
    'SEP': 15,  # PHOSPHOSERINE
    'THR': 16,
    'TPO': 16,  # PHOSPHOTHREONINE
    'TRP': 18,
    'TYR': 19,
    'PTR': 19,  # PHOSPHOTYROSINE
    'VAL': 17,
    'LLP': 20,  # NON STANDARD
    'ACE': 20,
    'UNK': 20,
    '_':   20,
}


class ProteinVocabulary(object):
    """Represents the 'vocabulary' of amino acids for encoding a protein sequence.

    Includes pad, sos, eos, and unknown characters as well as the 20 standard
    amino acids.
    """

    def __init__(self,
                 add_sos_eos=False,
                 include_unknown_char=False,
                 include_pad_char=True):
        self.include_unknown_char = include_unknown_char
        self.include_pad_char = include_pad_char
        self.pad_char = "_"  # Pad character
        self.unk_char = "?"  # unknown character
        self.sos_char = "<"  # SOS character
        self.eos_char = ">"  # EOS character

        self._char2int = dict()
        self._int2char = dict()

        # Extract the ordered list of 1-letter amino acid codes from the project-level
        # AA_MAP.
        self.stdaas = map(lambda x: x[0], sorted(list(AA_MAP.items()),
                                                 key=lambda x: x[1]))
        self.stdaas = "".join(filter(lambda x: len(x) == 1, self.stdaas))
        for aa in self.stdaas:
            self.add(aa)

        if include_pad_char:
            self.add(self.pad_char)
            self.pad_id = self[self.pad_char]
        else:
            self.pad_id = 0  # Implicit padding with all-zeros
        if include_unknown_char:
            self.add(self.unk_char)
        if add_sos_eos:
            self.add(self.sos_char)
            self.add(self.eos_char)
            self.sos_id = self[self.sos_char]
            self.eos_id = self[self.eos_char]

    def __getitem__(self, aa):
        if self.include_unknown_char:
            return self._char2int.get(aa, self._char2int[self.unk_char])
        else:
            return self._char2int.get(aa, self._char2int[self.pad_char])

    def __contains__(self, aa):
        return aa in self._char2int

    def __setitem__(self, key, value):
        raise ValueError('vocabulary is readonly')

    def __len__(self):
        return len(self._char2int)

    def __repr__(self):
        return f"ProteinVocabulary[size={len(self)}]"

    def int2char(self, idx):
        return self._int2char[idx]

    def int2chars(self, idx):
        return ONE_TO_THREE_LETTER_MAP[self._int2char[idx]]

    def add(self, aa):
        if aa not in self:
            aaid = self._char2int[aa] = len(self)
            self._int2char[aaid] = aa
            return aaid
        else:
            return self[aa]

    def str2ints(self, seq, add_sos_eos=True):
        if add_sos_eos:
            return [self["<"]] + [self[aa] for aa in seq] + [self[">"]]
        else:
            return [self[aa] for aa in seq]

    def ints2str(self, ints, include_sos_eos=False, exclude_pad=False):
        seq = ""
        for i in ints:
            c = self.int2char(i)
            if exclude_pad and c == self.pad_char:
                continue
            if include_sos_eos or (c not in [self.sos_char, self.eos_char, self.pad_char]):
                seq += c
        return seq


VOCAB = ProteinVocabulary()

def ids_to_embed_input(x):
    """ Returns the amino acid string input for calculating the ESM and MSA transformer embeddings
        Inputs:
        * x: any deeply nested list of integers that correspond with amino acid id
    """
    assert isinstance(x, list), 'input must be a list'
    id2aa = VOCAB._int2char
    out = []

    for el in x:
        if isinstance(el, list):
            out.append(ids_to_embed_input(el))
        elif isinstance(el, int):
            out.append(id2aa[el])
        else:
            raise TypeError('type must be either list or character')

    if all(map(lambda c: isinstance(c, str), out)):
        return None, ''.join(out)

    return out

def get_esm_embed(seq, embed_model, batch_converter, msa_data=None):
    """ Returns the ESM embeddings for a protein.
        Inputs:
        * seq: ( (b,) L,) tensor of ints (in sidechainnet int-char convention)
        * embed_model: ESM nn (see train_end2end.py for an example)
        * batch_converter: ESM batch converter (see train_end2end.py for an example)
        Outputs: tensor of (batch, n_seqs, L, embed_dim)
            * n_seqs: number of sequences in the MSA. 1 for ESM-1b
            * embed_dim: number of embedding dimensions. 1280 for ESM-1b
    """
    #  use ESM transformer
    device = seq.device
    repr_layer_num = 33
    max_seq_len = seq.shape[-1]
    embed_inputs = ids_to_embed_input(seq.cpu().tolist())

    batch_labels, batch_strs, batch_tokens = batch_converter([embed_inputs])  # batch_converter([embed_inputs])
    with torch.no_grad():
        results = embed_model(batch_tokens.to(device), repr_layers=[repr_layer_num], return_contacts=False)
    # index 0 is for start token. so take from 1 one
    token_reps = results["representations"][repr_layer_num][..., 1:max_seq_len+1, :].unsqueeze(dim=1)
    return token_reps

def get_esm_model(device):
    torch.hub.set_dir('/data/masters/datasets')
    embed_model, alphabet = torch.hub.load("facebookresearch/esm", "esm1b_t33_650M_UR50S")
    batch_converter = alphabet.get_batch_converter()
    embed_model = embed_model.to(device)
    return embed_model, batch_converter
