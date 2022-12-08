import torch
from torch import nn
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, radius_graph
# from dig.threedgraph.utils import xyz_to_dat
# from dig.threedgraph.method.spherenet.spherenet import emb, update_v, update_u
from einops import repeat
from torch.utils.checkpoint import checkpoint_sequential

from edmdock.nn.layers import E_GCL, SphereNetInit, SphereNetUpdateE, Block, Residual, PreNorm, EquivariantAttention, FeedForward, GuidedAttention, SelfAttention
from edmdock.utils import exists, default, ACTIVATIONS, swish


class EGNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, edge_dim, n_layers, activation='silu', norm=False, clamp=False, dropout=0.0, **kwargs):
        super(EGNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.norm = norm
        self.dropout = dropout
        self.activation = ACTIVATIONS[activation]
        kwargs = dict(output_nf=hidden_dim, hidden_nf=hidden_dim, edges_in_d=edge_dim, act_fn=self.activation, clamp=clamp, dropout=dropout)
        self.add_module('gnn_0', E_GCL(input_nf=input_dim, recurrent=False, **kwargs))
        for i in range(1, n_layers):
            self.add_module("gnn_%d" % i, E_GCL(input_nf=hidden_dim, recurrent=True, **kwargs))
        if self.norm:
            self.norm_fn = nn.LayerNorm(hidden_dim)

    def forward(self, h, edges, coords=None, batch=None, edge_attr=None):
        for i in range(self.n_layers):
            h, _, _ = self._modules["gnn_%d" % i](h, edges, coords, edge_attr=edge_attr)
        if self.norm:
            h = self.norm_fn(h)
        return h


class GCN(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, activation, **kwargs):
        super(GCN, self).__init__()
        self.input_dim = input_dim + 3
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.add_module('gnn_0', GCNConv(self.input_dim, self.hidden_dim))
        for i in range(1, n_layers):
            self.add_module("gnn_%d" % i, GCNConv(self.hidden_dim, self.hidden_dim))

    def forward(self, h, edges, coords=None, batch=None, edge_attr=None):
        h = torch.cat([h, coords], dim=3) ## TODO check
        for i in range(self.n_layers):
            h = self._modules["gnn_%d" % i](h, edges)
            h = self.activation(h)
        return h


class GAT(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, activation, **kwargs):
        super(GAT, self).__init__()
        self.input_dim = input_dim + 3
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.activation = activation
        self.add_module('gnn_0', GATConv(self.input_dim, self.hidden_dim))
        for i in range(1, n_layers):
            self.add_module("gnn_%d" % i, GATConv(self.hidden_dim, self.hidden_dim))

    def forward(self, h, edges, coords=None, batch=None, edge_attr=None):
        h = torch.cat([h, coords], dim=3) ## TODO check
        for i in range(self.n_layers):
            h = self._modules["gnn_%d" % i](h, edges)
            h = self.activation(h)
        return h


# class SphereNet(nn.Module):
#     r"""
#     The spherical message passing neural nn SphereNet from the
#     "Spherical Message Passing for 3D Graph Networks" <https://arxiv.org/abs/2102.05013>_ paper.
#
#     Args:
#         energy_and_force (bool, optional): If set to True, will predict energy and take the negative of the
#             derivative of the energy with respect to the atomic positions as predicted forces. (default: False)
#         cutoff (float, optional): Cutoff distance for interatomic interactions. (default: 5.0)
#         num_layers (int, optional): Number of building blocks. (default: 4)
#         hidden_channels (int, optional): Hidden embedding size. (default: 128)
#         out_channels (int, optional): Size of each output sample. (default: 1)
#         int_emb_size (int, optional): Embedding size used for interaction triplets. (default: 64)
#         basis_emb_size_dist (int, optional): Embedding size used in the basis transformation of distance. (default: 8)
#         basis_emb_size_angle (int, optional): Embedding size used in the basis transformation of angle. (default: 8)
#         basis_emb_size_torsion (int, optional): Embedding size used in the basis transformation of torsion. (default: 8)
#         out_emb_channels (int, optional): Embedding size used for atoms in the output block. (default: 256)
#         num_spherical (int, optional): Number of spherical harmonics. (default: 7)
#         num_radial (int, optional): Number of radial basis functions. (default: 6)
#         envelope_exponent (int, optional): Shape of the smooth cutoff. (default: 5)
#         num_before_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection.
#             (default: 1)
#         num_after_skip (int, optional): Number of residual layers in the interaction blocks before the skip connection.
#             (default: 2)
#         num_output_layers (int, optional): Number of linear layers for the output blocks. (default: 3)
#         act: (function, optional): The activation funtion. (default: swish)
#         output_init: (str, optional): The initialization fot the output. It could be GlorotOrthogonal and zeros.
#             (default: GlorotOrthogonal)
#     """
#
#     def __init__(
#             self,
#             energy_and_force=False,
#             cutoff=5.0,
#             num_layers=4,
#             input_channels=1,
#             hidden_channels=128,
#             out_channels=1,
#             int_emb_size=64,
#             basis_emb_size_dist=8,
#             basis_emb_size_angle=8,
#             basis_emb_size_torsion=8,
#             out_emb_channels=256,
#             num_spherical=7,
#             num_radial=6,
#             envelope_exponent=5,
#             num_before_skip=1,
#             num_after_skip=2,
#             num_output_layers=3,
#             act=swish,
#             output_init='GlorotOrthogonal',
#             use_node_features=True,
#             radius_graph=True,
#             norm=False,
#             use_u=False,
#             **kwargs
#     ):
#         super(SphereNet, self).__init__()
#         self.cutoff = cutoff
#         self.energy_and_force = energy_and_force
#         self.radius_graph = radius_graph
#         self.norm = norm
#         self.use_u = use_u
#
#         self.init_e = SphereNetInit(num_radial, input_channels, hidden_channels, act, norm, use_node_features)
#         self.init_v = update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
#         # self.init_u = update_u()
#         self.emb = emb(num_spherical, num_radial, cutoff, envelope_exponent)
#
#         self.update_vs = nn.ModuleList([
#             update_v(hidden_channels, out_emb_channels, out_channels, num_output_layers, act, output_init)
#             for _ in range(num_layers)
#         ])
#         self.update_es = nn.ModuleList([
#             SphereNetUpdateE(hidden_channels, int_emb_size, basis_emb_size_dist, basis_emb_size_angle, basis_emb_size_torsion,
#                      num_spherical, num_radial, num_before_skip, num_after_skip, norm, act)
#             for _ in range(num_layers)
#         ])
#         if use_u:
#             assert NotImplementedError
#             self.update_us = nn.ModuleList([update_u() for _ in range(num_layers)])
#
#         self.reset_parameters()
#
#     def reset_parameters(self):
#         self.init_e.reset_parameters()
#         self.init_v.reset_parameters()
#         self.emb.reset_parameters()
#         for update_e in self.update_es:
#             update_e.reset_parameters()
#         for update_v in self.update_vs:
#             update_v.reset_parameters()
#
#     def forward(self, h, edges, coords, batch, edge_attr=None):
#         if self.energy_and_force:
#             coords.requires_grad_()
#
#         if self.radius_graph:
#             # raise NotImplementedError
#             edge_index = radius_graph(coords, r=self.cutoff, batch=batch)
#
#         num_nodes = h.size(0)
#         # print(num_nodes)
#         dist, angle, torsion, i, j, idx_kj, idx_ji = xyz_to_dat(coords, edges, num_nodes, use_torsion=True)
#         emb = self.emb(dist, angle, torsion, idx_kj)
#         # print('rbf', torch.min(rbf), torch.max(rbf))
#         # Initialize edge, node, graph features
#         e = self.init_e(h, emb, i, j)
#         # print('e', torch.min(e[0]), torch.max(e[0]), torch.min(e[1]), torch.max(e[1]))
#         v = self.init_v(e, i)
#         # print('v', torch.min(v), torch.max(v))
#         # u = self.init_u(torch.zeros_like(scatter(v, batch, dim=0)), v, batch)  # scatter(v, batch, dim=0)
#         for idx, (update_e, update_v) in enumerate(zip(self.update_es, self.update_vs)):
#             e = update_e(e, emb, idx_kj, idx_ji)
#             v = update_v(e, i)
#             # v = self.norm(v)
#             # print(idx, torch.min(e[0]), torch.max(e[0]), torch.min(e[1]), torch.max(e[1]))
#             # print(idx, torch.min(v), torch.max(v))
#             # u = update_u(u, v, batch)  # u += scatter(v, batch, dim=0)
#         return v


class EnTransformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        rel_pos_emb = False,
        dim_head = 64,
        heads = 8,
        edge_dim = 0,
        coors_hidden_dim = 16,
        neighbors = 0,
        only_sparse_neighbors = False,
        num_adj_degrees = None,
        adj_dim = 0,
        valid_neighbor_radius = float('inf'),
        init_eps = 1e-3,
        norm_rel_coors = True,
        norm_coors_scale_init = 1.,
        use_cross_product = False,
        talking_heads = False,
        checkpoint = False,
        rotary_theta = 10000,
        rel_dist_cutoff = 5000,
        rel_dist_scale = 1e2,
        attn_dropout = 0.,
        ff_dropout = 0.,
        **kwargs
    ):
        super().__init__()
        assert dim_head >= 32, 'your dimension per head should be greater than 32 for rotary embeddings to work well'
        assert not (exists(num_adj_degrees) and num_adj_degrees < 1), 'make sure adjacent degrees is greater than 1'

        if only_sparse_neighbors:
            num_adj_degrees = default(num_adj_degrees, 1)

        self.num_adj_degrees = num_adj_degrees
        self.adj_emb = nn.Embedding(num_adj_degrees + 1, adj_dim) if exists(num_adj_degrees) and adj_dim > 0 else None
        adj_dim = adj_dim if exists(num_adj_degrees) else 0

        self.checkpoint = checkpoint
        self.layers = nn.ModuleList([])

        for ind in range(depth):
            self.layers.append(Block(
                Residual(PreNorm(dim, EquivariantAttention(dim = dim, dim_head = dim_head, heads = heads, coors_hidden_dim = coors_hidden_dim, edge_dim = (edge_dim + adj_dim),  neighbors = neighbors, only_sparse_neighbors = only_sparse_neighbors, valid_neighbor_radius = valid_neighbor_radius, init_eps = init_eps, rel_pos_emb = rel_pos_emb, norm_rel_coors = norm_rel_coors, norm_coors_scale_init = norm_coors_scale_init, use_cross_product = use_cross_product, talking_heads = talking_heads, rotary_theta = rotary_theta, rel_dist_cutoff = rel_dist_cutoff, rel_dist_scale = rel_dist_scale, dropout = attn_dropout))),
                Residual(PreNorm(dim, FeedForward(dim = dim, dropout = ff_dropout)))
            ))

    def forward(
        self,
        h,
        coords,
        edges = None,
        edge_attr = None,
        batch = None,
        mask = None,
        return_coor_changes = False,
        **kwargs
    ):
        b, n, _ = h.shape

        # adj_mat from edges
        edge_index = edges
        edges = edge_attr
        adj_mat = torch.zeros((n, n)).to(h.device)

        assert not (exists(adj_mat) and (not exists(self.num_adj_degrees) or self.num_adj_degrees == 0)), 'num_adj_degrees must be greater than 0 if you are passing in an adjacency matrix'

        if exists(self.num_adj_degrees):
            assert exists(adj_mat), 'adjacency matrix must be passed in (keyword argument adj_mat)'

            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat.clone(), 'i j -> b i j', b = b)

            adj_indices = adj_mat.clone().long()

            for ind in range(self.num_adj_degrees - 1):
                degree = ind + 2

                next_degree_adj_mat = (adj_mat.float() @ adj_mat.float()) > 0
                next_degree_mask = (next_degree_adj_mat.float() - adj_mat.float()).bool()
                adj_indices.masked_fill_(next_degree_mask, degree)
                adj_mat = next_degree_adj_mat.clone()

            if exists(self.adj_emb):
                adj_emb = self.adj_emb(adj_indices)
                edges = torch.cat((edges, adj_emb), dim = -1) if exists(edges) else adj_emb

        assert not (return_coor_changes and self.training), 'you must be eval mode in order to return coordinates'

        # go through layers

        coor_changes = [coords]
        inp = (h, coords, mask, edges, adj_mat)

        # if in training mode and checkpointing is designated, use checkpointing across blocks to save memory
        if self.training and self.checkpoint:
            inp = checkpoint_sequential(self.layers, len(self.layers), inp)
        else:
            # iterate through blocks
            for layer in self.layers:
                inp = layer(inp)
                coor_changes.append(inp[1]) # append coordinates for visualization

        # return

        feats, coors, *_ = inp

        if return_coor_changes:
            return feats, coors, coor_changes

        return feats, coors


class L2(nn.Module):
    def __init__(self, input_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim

    @staticmethod
    def forward(ligand_h, protein_h, edge_index):
        diff = ligand_h[edge_index[0]] - protein_h[edge_index[1]]
        # pred = torch.linalg.norm(diff, dim=1)
        # return torch.vstack([pred, torch.zeros_like(pred, device=pred.device)]).T
        # var = torch.abs(diff[:, 0]) #torch.sum(, dim=1)
        # mu = torch.linalg.norm(diff[:, 1:], dim=1)
        var = torch.sum(torch.abs(diff[:, :128]), dim=1)
        mu = torch.linalg.norm(diff[:, 128:], dim=1)
        return torch.vstack([mu, var]).T


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, activation, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.act = ACTIVATIONS[activation]
        self.norm = nn.BatchNorm1d(self.hidden_dim)
        self.inp = nn.Linear(self.input_dim, self.hidden_dim)
        for i in range(n_layers):
            self.add_module(f'hidden_{i}', nn.Linear(self.hidden_dim, self.hidden_dim))
        self.out = nn.Linear(self.hidden_dim, 25)

    def forward(self, ligand_h, protein_h, edge_index):
        inter_h = ligand_h[edge_index[0]] - protein_h[edge_index[1]]
        inter_h = self.act(self.inp(inter_h))
        for i in range(self.n_layers):
            inter_h = self.act(self._modules[f'hidden_{i}'](inter_h))
        inter_h = self.out(inter_h)
        inter_h = torch.vstack([inter_h[:, 0], torch.linalg.norm(inter_h[:, 1:], dim=1)]).T
        return inter_h


class MLP2(nn.Module):
    def __init__(self, input_dim, hidden_dim, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 4, 2),
            nn.ReLU()
        )

    def forward(self, ligand_h, protein_h, edge_index):
        h = ligand_h[edge_index[0]] - protein_h[edge_index[1]]
        return self.net(h)


class Attn(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers=1, activation=None, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        for i in range(n_layers):
            inp = self.input_dim if i == 0 else self.hidden_dim
            self.add_module(f'ligand_{i}', GuidedAttention(inp, self.hidden_dim))
            self.add_module(f'protein_{i}', GuidedAttention(inp, self.hidden_dim))

        if activation is not None:
            self.act = ACTIVATIONS[activation]

        self.out = nn.Linear(self.hidden_dim, 25)

    def forward(self, ligand_h, protein_h, edge_index):
        for i in range(self.n_layers):
            ligand_h = self._modules[f'ligand_{i}'](ligand_h, protein_h)
            protein_h = self._modules[f'protein_{i}'](protein_h, ligand_h)

        inter_h = ligand_h[edge_index[0]] - protein_h[edge_index[1]]
        inter_h = self.out(inter_h)
        inter_h = torch.vstack([torch.linalg.norm(inter_h[:, 1:], dim=1), inter_h[:, 0]]).T
        return inter_h


class Attn2(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, activation, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.act = ACTIVATIONS[activation]
        self.add_module('gat_0', GATConv(self.input_dim, self.hidden_dim))
        for i in range(1, n_layers):
            self.add_module("gat_%d" % i, GATConv(self.hidden_dim, self.hidden_dim))

    def forward(self, ligand_h, protein_h, edge_index):
        edges = torch.vstack([edge_index[0], edge_index[1] + len(ligand_h)])
        h = torch.cat([ligand_h, protein_h], dim=0)
        for i in range(self.n_layers):
            h = self.act(self._modules["gat_%d" % i](h, edges))
        ligand_h, protein_h = h.split_with_sizes((len(ligand_h), len(protein_h)))
        diff = ligand_h[edge_index[0]] - protein_h[edge_index[1]]
        var = torch.sum(torch.abs(diff[:, :128]), dim=1)
        mu = torch.linalg.norm(diff[:, 128:], dim=1)
        return torch.vstack([mu, var]).T


MOL_NETS = {'egnn': EGNN, 'gcn': GCN, 'gat': GAT, 'en-transformer': EnTransformer} # 'spherenet': SphereNet,
DIS_NETS = {'l2': L2, 'mlp': MLP, 'mlp2': MLP2, 'attn': Attn, 'attn2': Attn2}
ALL_NETS = {**MOL_NETS, **DIS_NETS}

def create_net(net_config):
    net_class = ALL_NETS[net_config['model'].lower()]
    return net_class(**net_config)
