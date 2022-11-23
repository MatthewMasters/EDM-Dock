import os
import argparse
from shutil import copy

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger

from edmdock import create_model, load_config, set_seed, create_run_path, load_dataset


def main():
    parser = argparse.ArgumentParser(description='edmdock')
    parser.add_argument('--config-path', type=str, help='path of config file', required=True)
    args = parser.parse_args()
    config = load_config(args.config_path)
    set_seed(config.seed)
    save_path = create_run_path(config.save_path)
    copy(args.config_path, os.path.join(save_path, 'config.yml'))

    model = create_model(config['model'])
    if config['resume_checkpoint'] is not None:
        model.load_state_dict(torch.load(config['resume_checkpoint'], map_location='cuda')['state_dict'])

    print(model)

    dl_kwargs = dict(batch_size=config['batch_size'], num_workers=config['num_workers'])
    print('Loading training set...')
    train_dl = load_dataset(config['data']['train_path'], config['data']['filename'], shuffle=True, **dl_kwargs)
    print('Loading validation set...')
    # valid_dl = load_dataset(config['data']['valid_path'], config['data']['filename'], shuffle=False, **dl_kwargs)

    trainer = Trainer(
        max_epochs=config.epochs,
        gpus=str(config.cuda),
        gradient_clip_val=config.grad_clip,
        logger=CSVLogger(save_path),
        accumulate_grad_batches=config.grad_acc_step,
        callbacks=ModelCheckpoint(save_path) #, monitor='val_loss'),
    )
    print('Begin training...')
    trainer.fit(model, train_dl) #, valid_dl)


if __name__ == '__main__':
    main()
    exit()
