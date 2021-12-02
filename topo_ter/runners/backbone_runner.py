import os

import numpy as np
import torch.nn as nn

from tensorboardX import SummaryWriter

from topo_ter.datasets.data_provider import DataProvider
from topo_ter.models.backbone import Backbone
from topo_ter.runners.runner import Runner


class BackboneRunner(Runner):
    def __init__(self, args):
        super(BackboneRunner, self).__init__(args)
        self.loss = nn.CrossEntropyLoss().to(self.output_dev)

    def load_dataset(self):
        self.dataset = DataProvider(
            data_path=self.args.data_path,
            dataset=self.args.dataset,
            perturbation_rate=self.args.perturbation_rate,
            stage='unsupervised'
        )

    def load_model(self):
        model = Backbone(
            in_channels=self.dataset.num_features,
            hidden_channels=self.args.hidden_channels,
            k=self.args.k
        )
        self.model['train'] = model.to(self.output_dev)

    def initialize_model(self):
        if self.args.backbone is not None:
            self.load_model_weights(self.model['train'], self.args.backbone)
            self.load_optimizer_weights(self.optimizer, self.args.backbone)

    def run(self):
        model_path = os.path.join(
            self.args.save_dir, f'{self.args.dataset}_best.pt'
        )
        best_epoch = -1
        best_loss = np.Inf
        for epoch in range(self.args.num_epochs):
            loss = self._train_backbone(epoch)
            self.print_log(
                'Train Unsupervised Epoch {}: Loss {:.5f}.'.format(epoch, loss)
            )
            if loss < best_loss:
                best_loss = loss
                best_epoch = epoch
                if self.args.save_model:
                    self.save_weights(self.model['train'], model_path)
            if epoch - best_epoch > self.args.patience:
                break
        # self.args.backbone = model_path
        # ClassifierRunner(self.args).run()

    def _train_backbone(self, epoch):
        self.model['train'].train()
        self.optimizer.zero_grad()
        # data
        data = self.dataset.next()
        x = data['x'].to(self.output_dev)
        edge1 = data['original_edge'].long().to(self.output_dev)
        edge2 = data['perturbed_edge'].long().to(self.output_dev)
        sampled = data['sampled_edge'].long().to(self.output_dev)
        edge_types = data['perturbation_type'].to(self.output_dev)
        # forward
        t_hat = self.model['train'](x, edge1, edge2, sampled)
        loss = self.loss(t_hat, edge_types)
        # backward
        loss.backward()
        self.optimizer.step()

        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/backbone_loss', loss.item(), epoch)
        return loss.item()
