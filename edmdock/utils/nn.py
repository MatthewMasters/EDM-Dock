import random

import torch
import numpy as np
from torch import nn

ACTIVATIONS = {'silu': nn.SiLU(), 'relu': nn.ReLU(), 'tanh': nn.Tanh()}


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print('set seed for random, numpy and torch')


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def unsorted_segment_sum(data, segment_ids, num_segments):
    """Custom PyTorch op to replicate TensorFlow's `unsorted_segment_sum`."""
    result_shape = (num_segments, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result.scatter_add_(0, segment_ids, data)
    return result


def unsorted_segment_mean(data, segment_ids, num_segments):
    result_shape = (num_segments, data.size(1))
    segment_ids = segment_ids.unsqueeze(-1).expand(-1, data.size(1))
    result = data.new_full(result_shape, 0)  # Init empty result tensor.
    count = data.new_full(result_shape, 0)
    result.scatter_add_(0, segment_ids, data)
    count.scatter_add_(0, segment_ids, torch.ones_like(data))
    return result / count.clamp(min=1)


def broadcat(tensors, dim=-1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]

    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))

    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all(
        [*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatenation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim=dim)


def batched_index_select(values, indices, dim=1):
    value_dims = values.shape[(dim + 1):]
    values_shape, indices_shape = map(lambda t: list(t.shape), (values, indices))
    indices = indices[(..., *((None,) * len(value_dims)))]
    indices = indices.expand(*((-1,) * len(indices_shape)), *value_dims)
    value_expand_len = len(indices_shape) - (dim + 1)
    values = values[(*((slice(None),) * dim), *((None,) * value_expand_len), ...)]

    value_expand_shape = [-1] * len(values.shape)
    expand_slice = slice(dim, (dim + value_expand_len))
    value_expand_shape[expand_slice] = indices.shape[expand_slice]
    values = values.expand(*value_expand_shape)

    dim += value_expand_len
    return values.gather(dim, indices)


def _random_angle(min_val, max_val):
    return torch.distributions.uniform.Uniform(min_val, max_val).sample()


def rotate_x(coords, rotation_angle):
    """Rotate the point cloud about the X-axis."""
    cost = torch.cos(rotation_angle)
    sint = torch.sin(rotation_angle)
    rotation_matrix = torch.tensor([[1.0, 0.0, 0.0], [0.0, cost, -sint], [0.0, sint, cost]], device=coords.device)
    rotated_data = torch.matmul(coords, rotation_matrix)
    return rotated_data


def rotate_y(coords, rotation_angle):
    """Rotate the point cloud about the Y-axis."""
    cost = torch.cos(rotation_angle)
    sint = torch.sin(rotation_angle)
    rotation_matrix = torch.tensor([[cost, 0.0, sint], [0.0, 1.0, 0.0], [-sint, 0.0, cost]], device=coords.device)
    rotated_data = torch.matmul(coords, rotation_matrix)
    return rotated_data


def rotate_z(coords, rotation_angle):
    """Rotate the point cloud about the Z-axis."""
    cost = torch.cos(rotation_angle)
    sint = torch.sin(rotation_angle)
    rotation_matrix = torch.tensor([[cost, -sint, 0.0], [sint, cost, 0.0], [0.0, 0.0, 1.0]], device=coords.device)
    rotated_data = torch.matmul(coords, rotation_matrix)
    return rotated_data


def random_rotate_x(coords, min_val=0.0, max_val=2.0 * np.pi):
    """Randomly rotate the point cloud about the X-axis."""
    return rotate_x(coords, _random_angle(min_val, max_val))


def random_rotate_y(coords, min_val=0.0, max_val=2.0 * np.pi):
    """Randomly rotate the point cloud about the Y-axis."""
    return rotate_y(coords, _random_angle(min_val, max_val))


def random_rotate_z(coords, min_val=0.0, max_val=2.0 * np.pi):
    """Randomly rotate the point cloud about the Z-axis."""
    return rotate_z(coords, _random_angle(min_val, max_val))


def random_rotate(coords):
    coords = random_rotate_x(coords)
    coords = random_rotate_y(coords)
    coords = random_rotate_z(coords)
    return coords


def swish(x):
    return x * x.sigmoid()


def get_optimizer(model, name='adam', lr=0.01):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    name = name.lower()
    if name == "adam":
        return torch.optim.Adam(parameters, lr=lr)
    elif name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    elif name == 'adadelta':
        return torch.optim.Adadelta(parameters, lr=lr)
    elif name == 'adagrad':
        return torch.optim.Adagrad(parameters, lr=lr)
    else:
        raise NotImplementedError('Optimizer not supported: %s' % name)
