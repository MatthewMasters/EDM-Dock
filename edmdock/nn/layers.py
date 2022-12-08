import torch
import torch.nn.functional as F
from torch import nn, einsum
from torch_geometric.nn.inits import glorot_orthogonal
from torch_scatter import scatter
from einops import rearrange, repeat

from edmdock.utils import exists, max_neg_value, unsorted_segment_sum, unsorted_segment_mean, batched_index_select, broadcat, swish


class E_GCL(nn.Module):
    """Graph Neural Net with global state and fixed number of nodes per graph."""

    def __init__(
            self,
            input_nf,
            output_nf,
            hidden_nf,
            edges_in_d=0,
            nodes_att_dim=0,
            act_fn=nn.ReLU(),
            recurrent=True,
            coords_weight=1.0,
            attention=False,
            clamp=False,
            norm_diff=False,
            tanh=False,
            dropout=0.0
    ):
        super(E_GCL, self).__init__()
        input_edge = input_nf * 2
        self.coords_weight = coords_weight
        self.recurrent = recurrent
        self.attention = attention
        self.norm_diff = norm_diff
        self.tanh = tanh
        edge_coords_nf = 1

        self.edge_mlp = nn.Sequential(
            nn.Linear(input_edge + edge_coords_nf + edges_in_d, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, hidden_nf),
            act_fn
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(hidden_nf + input_nf + nodes_att_dim, hidden_nf),
            act_fn,
            nn.Linear(hidden_nf, output_nf)
        )

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight, gain=0.001)

        self.clamp = clamp
        coord_mlp = [nn.Linear(hidden_nf, hidden_nf), act_fn, layer]
        if self.tanh:
            coord_mlp.append(nn.Tanh())
            self.coords_range = nn.Parameter(torch.ones(1)) * 3
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.att_mlp = nn.Sequential(
                nn.Linear(hidden_nf, 1),
                nn.Sigmoid()
            )

        self.dropout = nn.Dropout(dropout)

    def edge_model(self, source, target, radial, edge_attr):
        if edge_attr is None:  # Unused.
            out = torch.cat([source, target, radial], dim=1).float()
        else:
            out = torch.cat([source, target, radial, edge_attr], dim=1)

        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.att_mlp(out)
            out = out * att_val
        return out

    def node_model(self, x, edge_index, edge_attr, node_attr):
        row, col = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=x.size(0))
        if node_attr is not None:
            agg = torch.cat([x, agg, node_attr], dim=1).float()
        else:
            agg = torch.cat([x, agg], dim=1).float()
        out = self.node_mlp(agg)
        if self.recurrent:
            out = x + out
        return out, agg

    def coord_model(self, coord, edge_index, coord_diff, edge_feat):
        row, col = edge_index
        trans = coord_diff * self.coord_mlp(edge_feat)
        trans = torch.clamp(trans, min=-100, max=100)
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        coord += agg * self.coords_weight
        return coord

    def coord2radial(self, edge_index, coord):
        row, col = edge_index
        coord_diff = coord[row] - coord[col]
        radial = torch.sum(coord_diff ** 2, 1).unsqueeze(1)

        if self.norm_diff:
            norm = torch.sqrt(radial) + 1
            coord_diff = coord_diff / norm

        return radial, coord_diff

    def forward(self, h, edge_index, coord, edge_attr=None, node_attr=None):
        # edge_index should be bidirectional
        # print(edge_index.shape)
        row, col = edge_index
        radial, coord_diff = self.coord2radial(edge_index, coord)
        # print(h.shape, row)
        hr = h[row]
        hc = h[col]
        edge_feat = self.edge_model(hr, hc, radial, edge_attr)
        coord = self.coord_model(coord, edge_index, coord_diff, edge_feat)
        h, agg = self.node_model(h, edge_index, edge_feat, node_attr)
        h = self.dropout(h)
        return h, coord, edge_attr


class SphereNetInit(nn.Module):
    def __init__(self, num_radial, input_channels, hidden_channels, act=swish, norm=False, use_node_features=True):
        super(SphereNetInit, self).__init__()
        self.act = act
        self.lin_inp = nn.Linear(input_channels, hidden_channels)
        self.lin_rbf_0 = nn.Linear(num_radial, hidden_channels)
        self.lin = nn.Linear(3 * hidden_channels, hidden_channels)
        self.lin_rbf_1 = nn.Linear(num_radial, hidden_channels, bias=False)
        self.norm = norm
        if self.norm:
            self.norm_fn_e1 = nn.LayerNorm(hidden_channels)
            self.norm_fn_e2 = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin_rbf_0.reset_parameters()
        self.lin.reset_parameters()
        glorot_orthogonal(self.lin_rbf_1.weight, scale=2.0)

    def forward(self, x, emb, i, j):
        rbf, _, _ = emb
        # if self.use_node_features:
        #     x = self.emb(x)
        # else:
        #     x = self.node_embedding[None, :].expand(x.shape[0], -1)
        x = self.lin_inp(x)
        rbf0 = self.act(self.lin_rbf_0(rbf))
        e1 = self.act(self.lin(torch.cat([x[i], x[j], rbf0], dim=-1)))
        e2 = self.lin_rbf_1(rbf) * e1
        if self.norm:
            e1 = self.norm_fn_e1(e1)
            e2 = self.norm_fn_e1(e2)
        return e1, e2


class ResidualLayer(nn.Module):
    def __init__(self, hidden_channels, act=swish):
        super(ResidualLayer, self).__init__()
        self.act = act
        self.lin1 = nn.Linear(hidden_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        return x + self.act(self.lin2(self.act(self.lin1(x))))


class SphereNetUpdateE(torch.nn.Module):
    def __init__(
            self,
            hidden_channels,
            int_emb_size,
            basis_emb_size_dist,
            basis_emb_size_angle,
            basis_emb_size_torsion,
            num_spherical,
            num_radial,
            num_before_skip,
            num_after_skip,
            norm=False,
            act=swish
    ):
        super(SphereNetUpdateE, self).__init__()
        self.norm = norm
        self.act = act
        self.lin_rbf1 = nn.Linear(num_radial, basis_emb_size_dist, bias=False)
        self.lin_rbf2 = nn.Linear(basis_emb_size_dist, hidden_channels, bias=False)
        self.lin_sbf1 = nn.Linear(num_spherical * num_radial, basis_emb_size_angle, bias=False)
        self.lin_sbf2 = nn.Linear(basis_emb_size_angle, int_emb_size, bias=False)
        self.lin_t1 = nn.Linear(num_spherical * num_spherical * num_radial, basis_emb_size_torsion, bias=False)
        self.lin_t2 = nn.Linear(basis_emb_size_torsion, int_emb_size, bias=False)
        self.lin_rbf = nn.Linear(num_radial, hidden_channels, bias=False)

        self.lin_kj = nn.Linear(hidden_channels, hidden_channels)
        self.lin_ji = nn.Linear(hidden_channels, hidden_channels)

        self.lin_down = nn.Linear(hidden_channels, int_emb_size, bias=False)
        self.lin_up = nn.Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_before_skip)
        ])
        self.lin = nn.Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act)
            for _ in range(num_after_skip)
        ])

        if self.norm:
            self.norm_fn_e1 = nn.LayerNorm(hidden_channels)
            self.norm_fn_e2 = nn.LayerNorm(hidden_channels)
        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_t1.weight, scale=2.0)
        glorot_orthogonal(self.lin_t2.weight, scale=2.0)

        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)

        glorot_orthogonal(self.lin_down.weight, scale=2.0)
        glorot_orthogonal(self.lin_up.weight, scale=2.0)

        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)

    def forward(self, x, emb, idx_kj, idx_ji):
        rbf0, sbf, t = emb
        x1, _ = x

        x_ji = self.act(self.lin_ji(x1))
        x_kj = self.act(self.lin_kj(x1))

        rbf = self.lin_rbf1(rbf0)
        rbf = self.lin_rbf2(rbf)
        x_kj = x_kj * rbf

        x_kj = self.act(self.lin_down(x_kj))

        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)
        x_kj = x_kj[idx_kj] * sbf

        t = self.lin_t1(t)
        t = self.lin_t2(t)
        x_kj = x_kj * t

        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x1.size(0))
        x_kj = self.act(self.lin_up(x_kj))

        e1 = x_ji + x_kj
        for layer in self.layers_before_skip:
            e1 = layer(e1)
        e1 = self.act(self.lin(e1)) + x1
        for layer in self.layers_after_skip:
            e1 = layer(e1)
        e2 = self.lin_rbf(rbf0) * e1
        if self.norm:
            e1 = self.norm_fn_e1(e1)
            e2 = self.norm_fn_e1(e2)
        return e1, e2


class CoorsNorm(nn.Module):
    def __init__(self, eps=1e-8, scale_init=1.):
        super().__init__()
        self.eps = eps
        scale = torch.zeros(1).fill_(scale_init)
        self.scale = nn.Parameter(scale)

    def forward(self, coors):
        norm = coors.norm(dim=-1, keepdim=True)
        normed_coors = coors / norm.clamp(min=self.eps)
        return normed_coors * self.scale


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, feats, coors, **kwargs):
        feats = self.norm(feats)
        feats, coors = self.fn(feats, coors, **kwargs)
        return feats, coors


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, feats, coors, **kwargs):
        feats_out, coors_delta = self.fn(feats, coors, **kwargs)
        return feats + feats_out, coors + coors_delta


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(
            self,
            *,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 4 * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim)
        )

    def forward(self, feats, coors):
        return self.net(feats), 0


class EquivariantAttention(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=4,
            edge_dim=0,
            coors_hidden_dim=16,
            neighbors=0,
            only_sparse_neighbors=False,
            valid_neighbor_radius=float('inf'),
            init_eps=1e-3,
            rel_pos_emb=None,
            edge_mlp_mult=2,
            norm_rel_coors=True,
            norm_coors_scale_init=1.,
            use_cross_product=False,
            talking_heads=False,
            rotary_theta=10000,
            rel_dist_cutoff=5000,
            rel_dist_scale=1e2,
            dropout=0.
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.neighbors = neighbors
        self.only_sparse_neighbors = only_sparse_neighbors
        self.valid_neighbor_radius = valid_neighbor_radius

        attn_inner_dim = heads * dim_head
        self.heads = heads
        self.to_qkv = nn.Linear(dim, attn_inner_dim * 3, bias=False)
        self.to_out = nn.Linear(attn_inner_dim, dim)

        self.talking_heads = nn.Conv2d(heads, heads, 1, bias=False) if talking_heads else None

        self.edge_mlp = None
        has_edges = edge_dim > 0

        if has_edges:
            edge_input_dim = heads + edge_dim
            edge_hidden = edge_input_dim * edge_mlp_mult

            self.edge_mlp = nn.Sequential(
                nn.Linear(edge_input_dim, edge_hidden),
                nn.GELU(),
                nn.Linear(edge_hidden, heads)
            )

            self.coors_mlp = nn.Sequential(
                nn.GELU(),
                nn.Linear(heads, heads)
            )
        else:
            self.coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads)
            )

        self.coors_gate = nn.Sequential(
            nn.Linear(heads, heads),
            nn.Tanh()
        )

        self.use_cross_product = use_cross_product
        if use_cross_product:
            self.cross_coors_mlp = nn.Sequential(
                nn.Linear(heads, coors_hidden_dim),
                nn.GELU(),
                nn.Linear(coors_hidden_dim, heads * 2)
            )

        self.norm_rel_coors = CoorsNorm(scale_init=norm_coors_scale_init) if norm_rel_coors else nn.Identity()

        num_coors_combine_heads = (2 if use_cross_product else 1) * heads
        self.coors_combine = nn.Parameter(torch.randn(num_coors_combine_heads))

        self.rotary_emb = SinusoidalEmbeddings(dim_head // (2 if rel_pos_emb else 1), theta=rotary_theta)
        self.rotary_emb_seq = SinusoidalEmbeddings(dim_head // 2, theta=rotary_theta) if rel_pos_emb else None

        self.rel_dist_cutoff = rel_dist_cutoff
        self.rel_dist_scale = rel_dist_scale

        self.node_dropout = nn.Dropout(dropout)
        self.coor_dropout = nn.Dropout(dropout)

        self.init_eps = init_eps
        self.apply(self.init_)

    def init_(self, module):
        if type(module) in {nn.Linear}:
            nn.init.normal_(module.weight, std=self.init_eps)

    def forward(
            self,
            feats,
            coors,
            edges=None,
            mask=None,
            adj_mat=None
    ):
        b, n, d, h, num_nn, only_sparse_neighbors, valid_neighbor_radius, device = *feats.shape, self.heads, self.neighbors, self.only_sparse_neighbors, self.valid_neighbor_radius, feats.device

        assert not (only_sparse_neighbors and not exists(
            adj_mat)), 'adjacency matrix must be passed in if only_sparse_neighbors is turned on'

        if exists(mask):
            num_nodes = mask.sum(dim=-1)

        rel_coors = rearrange(coors, 'b i d -> b i () d') - rearrange(coors, 'b j d -> b () j d')
        rel_dist = rel_coors.norm(p=2, dim=-1)

        # calculate neighborhood indices

        nbhd_indices = None
        nbhd_masks = None
        nbhd_ranking = rel_dist.clone()

        if exists(adj_mat):
            if len(adj_mat.shape) == 2:
                adj_mat = repeat(adj_mat, 'i j -> b i j', b=b)

            self_mask = torch.eye(n, device=device).bool()
            self_mask = rearrange(self_mask, 'i j -> () i j')
            adj_mat.masked_fill_(self_mask, False)

            max_adj_neighbors = adj_mat.long().sum(dim=-1).max().item() + 1

            num_nn = max_adj_neighbors if only_sparse_neighbors else (num_nn + max_adj_neighbors)
            valid_neighbor_radius = 0 if only_sparse_neighbors else valid_neighbor_radius

            nbhd_ranking = nbhd_ranking.masked_fill(self_mask, -1.)
            nbhd_ranking = nbhd_ranking.masked_fill(adj_mat, 0.)

        if 0 < num_nn < n:
            # make sure padding does not end up becoming neighbors
            if exists(mask):
                ranking_mask = mask[:, :, None] * mask[:, None, :]
                nbhd_ranking = nbhd_ranking.masked_fill(~ranking_mask, 1e5)

            nbhd_values, nbhd_indices = nbhd_ranking.topk(num_nn, dim=-1, largest=False)
            nbhd_masks = nbhd_values <= valid_neighbor_radius

        # derive queries keys and values

        q, k, v = self.to_qkv(feats).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # calculate nearest neighbors

        i = j = n

        if exists(nbhd_indices):
            i, j = nbhd_indices.shape[-2:]
            nbhd_indices_with_heads = repeat(nbhd_indices, 'b n d -> b h n d', h=h)
            k = batched_index_select(k, nbhd_indices_with_heads, dim=2)
            v = batched_index_select(v, nbhd_indices_with_heads, dim=2)
            rel_dist = batched_index_select(rel_dist, nbhd_indices, dim=2)
            rel_coors = batched_index_select(rel_coors, nbhd_indices, dim=2)
        else:
            k = repeat(k, 'b h j d -> b h n j d', n=n)
            v = repeat(v, 'b h j d -> b h n j d', n=n)

        # prepare mask

        if exists(mask):
            q_mask = rearrange(mask, 'b i -> b () i ()')
            k_mask = repeat(mask, 'b j -> b i j', i=n)

            if exists(nbhd_indices):
                k_mask = batched_index_select(k_mask, nbhd_indices, dim=2)

            k_mask = rearrange(k_mask, 'b i j -> b () i j')

            mask = q_mask * k_mask

            if exists(nbhd_masks):
                mask &= rearrange(nbhd_masks, 'b i j -> b () i j')

        # generate and apply rotary embeddings

        rot_null = torch.zeros_like(rel_dist)

        q_pos_emb_rel_dist = self.rotary_emb(torch.zeros(n, device=device))

        rel_dist_to_rotate = (rel_dist * self.rel_dist_scale).clamp(max=self.rel_dist_cutoff)
        k_pos_emb_rel_dist = self.rotary_emb(rel_dist_to_rotate)

        q_pos_emb = rearrange(q_pos_emb_rel_dist, 'i d -> () () i d')
        k_pos_emb = rearrange(k_pos_emb_rel_dist, 'b i j d -> b () i j d')

        if exists(self.rotary_emb_seq):
            pos_emb = self.rotary_emb_seq(torch.arange(n, device=device))

            q_pos_emb_seq = rearrange(pos_emb, 'n d -> () () n d')
            k_pos_emb_seq = rearrange(pos_emb, 'n d -> () () n () d')

            q_pos_emb = broadcat((q_pos_emb, q_pos_emb_seq), dim=-1)
            k_pos_emb = broadcat((k_pos_emb, k_pos_emb_seq), dim=-1)

        q = apply_rotary_pos_emb(q, q_pos_emb)
        k = apply_rotary_pos_emb(k, k_pos_emb)
        v = apply_rotary_pos_emb(v, k_pos_emb)

        # calculate inner product for queries and keys

        qk = einsum('b h i d, b h i j d -> b h i j', q, k) * (self.scale if not exists(edges) else 1)

        # add edge information and pass through edges MLP if needed

        if exists(edges):
            if exists(nbhd_indices):
                edges = batched_index_select(edges, nbhd_indices, dim=2)

            qk = rearrange(qk, 'b h i j -> b i j h')
            qk = torch.cat((qk, edges), dim=-1)
            qk = self.edge_mlp(qk)
            qk = rearrange(qk, 'b i j h -> b h i j')

        # coordinate MLP and calculate coordinate updates

        coors_mlp_input = rearrange(qk, 'b h i j -> b i j h')
        coor_weights = self.coors_mlp(coors_mlp_input)

        if exists(mask):
            mask_value = max_neg_value(coor_weights)
            coor_mask = repeat(mask, 'b () i j -> b i j ()')
            coor_weights.masked_fill_(~coor_mask, mask_value)

        coor_weights = coor_weights - coor_weights.amax(dim=-2, keepdim=True).detach()
        coor_attn = coor_weights.softmax(dim=-2)
        coor_attn = self.coor_dropout(coor_attn)

        rel_coors_sign = self.coors_gate(coors_mlp_input)
        rel_coors_sign = rearrange(rel_coors_sign, 'b i j h -> b i j () h')

        if self.use_cross_product:
            rel_coors_i = repeat(rel_coors, 'b n i c -> b n (i j) c', j=j)
            rel_coors_j = repeat(rel_coors, 'b n j c -> b n (i j) c', i=j)

            cross_coors = torch.cross(rel_coors_i, rel_coors_j, dim=-1)

            cross_coors = self.norm_rel_coors(cross_coors)
            cross_coors = repeat(cross_coors, 'b i j c -> b i j c h', h=h)

        rel_coors = self.norm_rel_coors(rel_coors)
        rel_coors = repeat(rel_coors, 'b i j c -> b i j c h', h=h)

        rel_coors = rel_coors * rel_coors_sign

        # cross product

        if self.use_cross_product:
            cross_weights = self.cross_coors_mlp(coors_mlp_input)

            cross_weights = rearrange(cross_weights, 'b i j (h n) -> b i j h n', n=2)
            cross_weights_i, cross_weights_j = cross_weights.unbind(dim=-1)

            cross_weights = rearrange(cross_weights_i, 'b n i h -> b n i () h') + rearrange(cross_weights_j,
                                                                                            'b n j h -> b n () j h')

            if exists(mask):
                cross_mask = (coor_mask[:, :, :, None, :] & coor_mask[:, :, None, :, :])
                cross_weights = cross_weights.masked_fill(~cross_mask, mask_value)

            cross_weights = rearrange(cross_weights, 'b n i j h -> b n (i j) h')
            cross_attn = cross_weights.softmax(dim=-2)

        # aggregate and combine heads for coordinate updates

        rel_out = einsum('b i j h, b i j c h -> b i c h', coor_attn, rel_coors)

        if self.use_cross_product:
            cross_out = einsum('b i j h, b i j c h -> b i c h', cross_attn, cross_coors)
            rel_out = torch.cat((rel_out, cross_out), dim=-1)

        coors_out = einsum('b n c h, h -> b n c', rel_out, self.coors_combine)

        # derive attention

        sim = qk.clone()

        if exists(mask):
            mask_value = max_neg_value(sim)
            sim.masked_fill_(~mask, mask_value)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)
        attn = self.node_dropout(attn)

        if exists(self.talking_heads):
            attn = self.talking_heads(attn)

        # weighted sum of values and combine heads

        out = einsum('b h i j, b h i j d -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, coors_out


class Block(nn.Module):
    def __init__(self, attn, ff):
        super().__init__()
        self.attn = attn
        self.ff = ff

    def forward(self, inp, coor_changes=None):
        feats, coors, mask, edges, adj_mat = inp
        feats, coors = self.attn(feats, coors, edges=edges, mask=mask, adj_mat=adj_mat)
        feats, coors = self.ff(feats, coors)
        return feats, coors, mask, edges, adj_mat


class SinusoidalEmbeddings(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        inv_freq = 1. / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, t):
        freqs = t[..., None].float() * self.inv_freq[None, :]
        freqs = repeat(freqs, '... d -> ... (d r)', r=2)
        return freqs


class SelfAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.to_q = nn.Linear(input_dim, hidden_dim)
        self.to_k = nn.Linear(input_dim, hidden_dim)
        self.to_v = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, x):
        q, k, v = self.to_q(x), self.to_k(x), self.to_v(x)
        out, _ = self.attn(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
        return out.squeeze()



class GuidedAttention(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_heads=4):
        super().__init__()
        self.to_q = nn.Linear(input_dim, hidden_dim)
        self.to_k = nn.Linear(input_dim, hidden_dim)
        self.to_v = nn.Linear(input_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads)

    def forward(self, query, guide):
        q, k, v = self.to_q(query), self.to_k(guide), self.to_v(guide)
        out, _ = self.attn(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1))
        return out.squeeze()




def rotate_half(x):
    x = rearrange(x, '... (d j) -> ... d j', j=2)
    x1, x2 = x.unbind(dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    rot_dim = freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)
