import os
import json
import pickle
import yaml
from easydict import EasyDict

import numpy as np


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def load_json(filepath):
    with open(filepath, 'r') as fi:
        d = json.load(fi)
    return d


def save_json(filepath, d):
    with open(filepath, 'w') as fo:
        json.dump(d, fo)


def load_pickle(filepath):
    with open(filepath, "rb") as fo:
        d = pickle.load(fo)
    return d


def save_pickle(filepath, d):
    with open(filepath, "wb") as fo:
        pickle.dump(d, fo)


def load_yaml(filepath):
    with open(filepath, 'r') as f:
        d = yaml.safe_load(f)
    return d


def save_yaml(filepath, d):
    with open(filepath, 'w') as fo:
        yaml.dump(d, fo)


def load_config(filepath):
    return EasyDict(load_yaml(filepath))


def create_run_path(checkpoint_path, name='run'):
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    def make_path(_i):
        return os.path.join(checkpoint_path, f'{name}_{_i:04d}')

    i = 0
    path = make_path(i)
    while os.path.exists(path):
        i += 1
        path = make_path(i)

    os.mkdir(path)
    return path


def get_last_checkpoint(run_path):
    filelist = [f for f in os.listdir(run_path) if f.endswith('.ckpt')]
    idx = np.argmax([f.split('.')[0].split('=')[-1] for f in filelist])
    checkpoint_path = os.path.join(run_path, filelist[idx])
    return checkpoint_path
