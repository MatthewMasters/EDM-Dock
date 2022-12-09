import os
import time
import argparse

from glob import glob
from multiprocessing import Pool

from edmdock.utils.feats import generate_pocket_multichain, generate_simple


def prepare(dataset_path, num_workers=1):
    print('Starting Preparation...')
    paths = list(glob(os.path.join(dataset_path, '*')))
    pool = Pool(processes=num_workers)

    # Select residues inside box
    print('Starting step 1: box')
    pool.map(generate_pocket_multichain, paths)

    # Generate features
    print('Starting step 2: features')
    pool.map(generate_simple, paths)

    # Review completeness
    req_files = ['simple.pkl']
    paths_n = len(paths)
    for req_file in req_files:
        req_file_n = len(glob(os.path.join(dataset_path, '*', req_file)))
        print(f'{req_file.ljust(10)}\t{str(req_file_n).zfill(5)}\t{req_file_n / paths_n:0.2f}')


def main():
    parser = argparse.ArgumentParser(description='confgf')
    parser.add_argument('--dataset_path', type=str, help='path of dataset', required=True)
    parser.add_argument('--num_workers', type=int, default=1, help='number of cpu workers to use', required=False)
    args = parser.parse_args()

    prepare(args.dataset_path, args.num_workers)


if __name__ == '__main__':
    main()
    exit()
