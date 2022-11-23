import torch
from torch import nn
import pytorch_lightning as pl

from .networks import create_net
from ..utils import get_optimizer


class Model(pl.LightningModule):
    def __init__(self, ligand_net, protein_net, distance_net, optimizer):
        super(Model, self).__init__()
        self.ligand_emb = nn.Embedding(9, 128)
        self.ligand_edge_emb = nn.Embedding(5, 16)
        self.pocket_emb = nn.Embedding(21, 128)
        self.ligand_net = ligand_net
        self.protein_net = protein_net
        self.distance_net = distance_net
        self.ligand_out = nn.Linear(self.ligand_net.hidden_dim, self.distance_net.input_dim)
        self.protein_out = nn.Linear(self.protein_net.hidden_dim, self.distance_net.input_dim)
        self.optimizer = optimizer
        self.mae = nn.MSELoss()

    def forward(self, batch):
        ligand_emb = self.ligand_emb(batch.ligand_types)
        ligand_features = torch.cat([ligand_emb, batch.ligand_features], dim=1)
        ligand_edges = self.ligand_edge_emb(batch.ligand_edge_types)
        pocket_features = self.pocket_emb(batch.pocket_types)
        ligand_h = self.ligand_net(ligand_features, batch.ligand_edge_index, batch.ligand_pos, batch.ligand_batch, ligand_edges)
        protein_h = self.protein_net(pocket_features, batch.pocket_edge_index, batch.pocket_pos, batch.pocket_batch)
        ligand_h = self.ligand_out(ligand_h)
        protein_h = self.protein_out(protein_h)
        distances = self.distance_net(ligand_h, protein_h, batch.inter_edge_index)
        return distances

    def loss(self, pred, target):
        mu, log_scale = pred.T
        scale = torch.log(log_scale + 1.01)
        dist = torch.distributions.normal.Normal(mu, scale)
        loss = -torch.mean(dist.log_prob(target))
        return loss

    def training_step(self, batch, batch_idx):
        target = batch.dis_gt
        pred = self.forward(batch)
        loss = self.loss(pred, target)
        metric = self.mae(pred[..., 0], target)
        self.log('train_mae', metric, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def training_epoch_end(self, outputs):
        loss = torch.stack([o['loss'] for o in outputs]).mean().item()
        self.log('train_loss', loss)

    def validation_step(self, batch, batch_idx):
        target = batch.dis_gt
        pred = self.forward(batch)
        loss = self.loss(pred, target)
        metric = self.mae(pred[..., 0], target)
        return pred, target, loss, metric

    def validation_epoch_end(self, outputs):
        preds, targets, losses, metrics = zip(*outputs)
        loss = torch.stack(losses).mean().item()
        metric = torch.stack(metrics).mean().item()
        self.log('val_loss', loss)
        self.log('val_mae', metric)

    def predict_step(self, batch, batch_idx, dataloader_idx):
        target = batch.dis_gt
        pred = self.forward(batch.clone())
        loss = self.loss(pred, target)
        return pred, target, loss, batch

    def configure_optimizers(self):
        optimizer = get_optimizer(self, **self.optimizer)
        return optimizer
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[250], gamma=0.1)
        # return [optimizer], [scheduler]


def create_model(model_config):
    ligand_net = create_net(model_config['ligand_net'])
    protein_net = create_net(model_config['protein_net'])
    interaction_net = create_net(model_config['interaction_net'])
    model = Model(ligand_net, protein_net, interaction_net, model_config['optimizer'])
    return model
