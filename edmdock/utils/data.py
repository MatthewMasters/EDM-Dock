import os
from multiprocessing import Pool

import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, Batch
from torch.utils.data import Sampler
from torch import Tensor
from torch_sparse import SparseTensor, cat
from torch.utils.data import DataLoader
from glob import glob
from tqdm import tqdm

from .utils import load_pickle


class PairData(Data):
    def __init__(
        self,
        key,
        ligand_features=None,
        bond_types=None,
        pocket_features=None,
        pocket_types=None,
        ligand_types=None,
        docked_pos=None,
        ligand_pos=None,
        pocket_pos=None,
        ligand_edge_types=None,
        ligand_edge_index=None,
        pocket_edge_index=None,
        inter_edge_index=None,
        dis_gt=None,
        **kwargs
    ):
        super().__init__()
        self.key = key
        self.ligand_features = ligand_features
        self.bond_types = bond_types
        self.pocket_features = pocket_features
        self.ligand_types = ligand_types
        self.pocket_types = pocket_types
        self.docked_pos = docked_pos
        self.ligand_pos = ligand_pos
        self.pocket_pos = pocket_pos
        self.ligand_edge_types = ligand_edge_types
        self.ligand_edge_index = ligand_edge_index
        self.pocket_edge_index = pocket_edge_index
        self.inter_edge_index = inter_edge_index
        self.dis_gt = dis_gt
        self.num_ligand_nodes = len(ligand_pos)
        self.num_pocket_nodes = len(pocket_pos)
        for key, item in kwargs.items():
            self[key] = item

    def __inc__(self, key, value):
        if key == 'ligand_edge_index':
            return self.num_ligand_nodes
        elif key == 'pocket_edge_index':
            return self.num_pocket_nodes
        elif key == 'inter_edge_index':
            return torch.tensor([[self.num_ligand_nodes], [self.num_pocket_nodes]])
        else:
            return super().__inc__(key, value)

    @classmethod
    def from_dict(cls, dictionary):
        r"""Creates a data object from a python dictionary."""
        data = cls(**dictionary)

        if torch_geometric.is_debug_enabled():
            data.debug()

        return data


class PairBatch(Batch):
    def __init__(self, batch=None, ptr=None, **kwargs):
        super(Batch, self).__init__(batch, ptr, **kwargs)

    @classmethod
    def from_data_list(cls, data_list, follow_batch=None, exclude_keys=None):
        r"""Constructs a batch object from a python list holding
        :class:`torch_geometric.data.Data` objects.
        The assignment vector :obj:`batch` is created on the fly.
        Additionally, creates assignment batch vectors for each key in
        :obj:`follow_batch`.
        Will exclude any keys given in :obj:`exclude_keys`."""
        if follow_batch is None:
            follow_batch = []
        if exclude_keys is None:
            exclude_keys = []

        keys = list(set(data_list[0].keys) - set(exclude_keys))
        assert 'batch' not in keys and 'ptr' not in keys

        batch = cls()
        for key in data_list[0].__dict__.keys():
            if key[:2] != '__' and key[-2:] != '__':
                batch[key] = None

        batch.__num_graphs__ = len(data_list)
        batch.__data_class__ = data_list[0].__class__
        for key in keys + ['pocket_batch', 'ligand_batch']:
            batch[key] = []
        batch['ptr'] = [0]

        device = None
        slices = {key: [0] for key in keys}
        cumsum = {key: [0] for key in keys}
        cat_dims = {}
        num_nodes_list = []
        for i, data in enumerate(data_list):
            for key in keys:
                item = data[key]

                # Increase values by `cumsum` value.
                cum = cumsum[key][-1]
                if isinstance(item, Tensor) and item.dtype != torch.bool:
                    if not isinstance(cum, int) or cum != 0:
                        item = item + cum
                elif isinstance(item, SparseTensor):
                    value = item.storage.value()
                    if value is not None and value.dtype != torch.bool:
                        if not isinstance(cum, int) or cum != 0:
                            value = value + cum
                        item = item.set_value(value, layout='coo')
                elif isinstance(item, (int, float)):
                    item = item + cum

                # Gather the size of the `cat` dimension.
                size = 1
                cat_dim = data.__cat_dim__(key, data[key])
                # 0-dimensional tensors have no dimension along which to
                # concatenate, so we set `cat_dim` to `None`.
                if isinstance(item, Tensor) and item.dim() == 0:
                    cat_dim = None
                cat_dims[key] = cat_dim

                # Add a batch dimension to items whose `cat_dim` is `None`:
                if isinstance(item, Tensor) and cat_dim is None:
                    cat_dim = 0  # Concatenate along this new batch dimension.
                    item = item.unsqueeze(0)
                    device = item.device
                elif isinstance(item, Tensor):
                    size = item.size(cat_dim)
                    device = item.device
                elif isinstance(item, SparseTensor):
                    size = torch.tensor(item.sizes())[torch.tensor(cat_dim)]
                    device = item.device()

                batch[key].append(item)  # Append item to the attribute list.

                slices[key].append(size + slices[key][-1])
                inc = data.__inc__(key, item)
                if isinstance(inc, (tuple, list)):
                    inc = torch.tensor(inc)
                cumsum[key].append(inc + cumsum[key][-1])

                if key in follow_batch:
                    if isinstance(size, Tensor):
                        for j, size in enumerate(size.tolist()):
                            tmp = f'{key}_{j}_batch'
                            batch[tmp] = [] if i == 0 else batch[tmp]
                            batch[tmp].append(
                                torch.full((size,), i, dtype=torch.long,
                                           device=device))
                    else:
                        tmp = f'{key}_batch'
                        batch[tmp] = [] if i == 0 else batch[tmp]
                        batch[tmp].append(
                            torch.full((size,), i, dtype=torch.long,
                                       device=device))

            if hasattr(data, '__num_nodes__'):
                num_nodes_list.append(data.__num_nodes__)
            else:
                num_nodes_list.append(None)

            # num_nodes = data.num_nodes
            # if num_nodes is not None:
            #     item = torch.full((num_nodes,), i, dtype=torch.long,
            #                       device=device)
            #     batch.batch.append(item)
            #     batch.ptr.append(batch.ptr[-1] + num_nodes)
            num_pocket_nodes = len(data.pocket_pos)
            num_ligand_nodes = len(data.ligand_pos)
            batch['pocket_batch'].append(torch.full((num_pocket_nodes,), i, dtype=torch.long, device=device))
            batch['ligand_batch'].append(torch.full((num_ligand_nodes,), i, dtype=torch.long, device=device))

        # batch.batch = None if len(batch.batch) == 0 else batch.batch
        batch.ptr = None if len(batch.ptr) == 1 else batch.ptr
        batch.__slices__ = slices
        batch.__cumsum__ = cumsum
        batch.__cat_dims__ = cat_dims
        batch.__num_nodes_list__ = num_nodes_list

        ref_data = data_list[0]
        for key in batch.keys:
            items = batch[key]
            item = items[0]
            cat_dim = ref_data.__cat_dim__(key, item)
            cat_dim = 0 if cat_dim is None else cat_dim
            if isinstance(item, Tensor):
                batch[key] = torch.cat(items, cat_dim)
            elif isinstance(item, SparseTensor):
                batch[key] = cat(items, cat_dim)
            elif isinstance(item, (int, float)):
                batch[key] = torch.tensor(items)

        if torch_geometric.is_debug_enabled():
            batch.debug()

        return batch.contiguous()


class PairCollater(object):
    def __init__(self, follow_batch=[], exclude_keys=[]):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    def __call__(self, batch):
        return PairBatch.from_data_list(batch, self.follow_batch, self.exclude_keys)


class BalancedSampler(Sampler):
    def __init__(self, index_path, dataset):
        super(BalancedSampler, self).__init__(dataset)
        self.ligand_index = load_pickle(index_path)
        self.ligand_keys = list(self.ligand_index.keys())
        self.num_ligands = len(self.ligand_index)
        self.num_docked = len(dataset)

    def __iter__(self):
        ligand_keys = np.random.choice(self.ligand_keys, size=self.num_ligands, replace=False)
        for key in ligand_keys:
            yield np.random.choice(self.ligand_index[key])

    def __len__(self):
        return self.num_ligands


class LogisticTransform:
    def __init__(self):
        pass

    def forward(self, x):
        return 10 * (2 / (1 + np.exp(-x / 5)) - 1)  # equivalent to 10*tanh(x/10) but faster

    def reverse(self, y):
        return 5 * np.log((-y - 10) / (y - 10))


class ParabolicTransform:
    def __init__(self):
        self.const = np.sqrt(30)

    def forward(self, x):
        return 2 * (1 - np.clip(x, 0.0, 30.0) / 60) * x

    def reverse(self, y):
        return 30 - self.const * np.sqrt(30 - y)



def load_dataset(dataset_path, filename, batch_size=1, num_workers=1, shuffle=True, skip_keys=None, n=None, skip_n=None):
    dataloader_kwargs = dict(
        collate_fn=PairCollater(),
        pin_memory=False,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle
    )

    iter_ = list(glob(os.path.join(dataset_path, '*', filename)))

    if skip_keys is not None:
        iter_ = [path for path in iter_ if path.split('/')[-2] not in skip_keys]
    if skip_n is not None:
        iter_ = iter_[skip_n:]
    if n is not None:
        iter_ = iter_[:n]

    dataset = []
    for path in tqdm(iter_):
        data = load_pickle(path)
        dataset.append(data)

    dl = DataLoader(dataset, **dataloader_kwargs)
    return dl
