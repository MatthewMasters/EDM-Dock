import os
import time
import argparse
from multiprocessing import Pool

import torch
import pandas as pd
from rdkit import Chem
from pytorch_lightning import Trainer
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix

from edmdock.utils.dock import Minimizer
from edmdock import create_model
from edmdock import load_config, set_seed, get_last_checkpoint, load_dataset
from edmdock import load_ligand, write_input, write_xyz, run_dgsol, get_results, c_to_d, align_coords, get_rmsd


def prepare_inputs_single(preds, batches):
    inputs = [
        (
            batch.key[0],
            pred.cpu().numpy(),
            *[getattr(batch, f'{k}_pos').cpu().numpy() for k in ['docked', 'pocket']])
        for pred, batch in zip(preds, batches)
    ]
    return inputs


def prepare_inputs_multi(preds, batches):
    inputs = []
    for pred, batch in zip(preds, batches):
        nmc = 0
        nc = 0
        mc = 0
        pred = pred.detach().cpu().numpy()
        docked_pos = batch.docked_pos.detach().cpu().numpy()
        pocket_pos = batch.pocket_pos.detach().cpu().numpy()
        print(pred.shape)
        for key, n, m in zip(batch.key, batch.num_ligand_nodes, batch.num_pocket_nodes):
            nm = n * m
            data = (
                key,
                pred[nmc:nmc + nm],
                docked_pos[nc:nc + n],
                pocket_pos[mc:mc + m],
            )
            nc += n
            mc += m
            nmc += n * m
            inputs.append(data)
    return inputs


def run_docking(inp):
    key, pred, docked_coords, pocket_coords = inp
    ligand_n = len(docked_coords)
    pocket_n = len(pocket_coords)
    mu, var = pred.T
    mu = mu.reshape(ligand_n, pocket_n)
    var = var.reshape(ligand_n, pocket_n)
    path = os.path.join(config['data']['test_path'], key)
    ligand_mol = load_ligand(os.path.join(path, 'ligand.sdf'))
    ligand_bm = GetMoleculeBoundsMatrix(ligand_mol)
    pocket_dm = c_to_d(pocket_coords)
    inp_path, out_path, sum_path = (os.path.join(results_path, f'{key}.{ext}') for ext in ['inp', 'out', 'sum'])

    write_input(inp_path, mu, var, ligand_bm, pocket_dm, k=dock_config['k'])
    run_dgsol(inp_path, out_path, sum_path, n_sol=dock_config['n_sol'])
    coords = get_results(out_path, sum_path, ligand_n, pocket_n)
    recon_pocket_coords, recon_ligand_coords = align_coords(coords, ligand_n, pocket_coords)
    rmsd = get_rmsd(docked_coords, recon_ligand_coords)

    # Update ligand positions
    for i, coord in enumerate(recon_ligand_coords):
        ligand_mol.GetConformer(0).SetAtomPosition(i, coord.tolist())
    rdkitmolh = Chem.AddHs(ligand_mol, addCoords=True)
    Chem.AssignAtomChiralTagsFromStructure(rdkitmolh)

    write_xyz(os.path.join(results_path, f'{key}_recon.xyz'), coords)
    write_xyz(os.path.join(results_path, f'{key}_docked.xyz'), recon_ligand_coords)
    Chem.MolToPDBFile(rdkitmolh, os.path.join(results_path, f'{key}_docked.pdb'))

    if dock_config['minimize']:
        # try:
        min_coords = minimizer.minimize(path, rdkitmolh, recon_pocket_coords, mu, var)
        min_rmsd = get_rmsd(docked_coords, min_coords)
        # except:
        #     min_rmsd = rmsd
        out = (key, rmsd, min_rmsd)
    else:
        out = (key, rmsd)
    print(out)
    return out


if __name__ == '__main__':
    t0 = time.time()
    parser = argparse.ArgumentParser(description='edmdock')
    parser.add_argument('--run-path', type=str, help='path of saved run (requires config and weights)', required=True)
    args = parser.parse_args()
    config_path = os.path.join(args.run_path, 'config.yml')
    weight_path = get_last_checkpoint(args.run_path)
    print(f'Using weights from... {weight_path}')
    results_path = os.path.join(args.run_path, 'results')
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    config = load_config(config_path)
    data_config, model_config, dock_config = config['data'], config['model'], config['dock']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(config['cuda'])
    set_seed(config.seed)

    model = create_model(model_config)
    model.load_state_dict(torch.load(weight_path)['state_dict']) #, map_location='cuda'
    model.eval()

    batch_size = 1
    dl_kwargs = dict(batch_size=batch_size, num_workers=config['num_workers'])
    print('Loading test set...')
    # TODO: tmp
    # all_keys = pd.read_csv('/data/masters/projects/EDM-Dock/dev_scripts/our_crossdock.csv')['key'].values
    # skip_keys = [key for key in all_keys if os.path.exists(f'/data/masters/datasets/edm-dock-dataset-simple/test/disco/{key}/{key}_min.pdb')]
    # print(len(skip_keys))
    # skip_keys = [key for key in all_keys if 'BACE1' not in key] # , skip_keys=skip_keys
    skip_keys = []
    test_dl = load_dataset(data_config['test_path'], data_config['filename'], skip_keys=skip_keys, n=500, skip_n=0, shuffle=False, **dl_kwargs)

    trainer = Trainer(gpus=config['cuda'])
    outputs = trainer.predict(model, test_dl)
    print(outputs)
    preds, targets, losses, batches = zip(*outputs)
    inputs = prepare_inputs_single(preds, batches) if batch_size == 1 else prepare_inputs_multi(preds, batches)

    columns = ['key', 'rmsd']
    if dock_config['minimize']:
        minimizer = Minimizer()
        columns += ['rmsd_min', 'energy']
        data = []
        for inp in inputs:
            out = run_docking(inp)
            data.append(out)
    else:
        pool = Pool(processes=config['num_workers'])
        data = pool.map(run_docking, inputs)
    data = pd.DataFrame(data, columns=columns)
    data.to_csv(os.path.join(results_path, f'results.csv'), index=False)
    print(data['rmsd'].describe())


    t1 = time.time()
    t = t1 - t0
    print(f'Total Time: {t:.1f} seconds')
