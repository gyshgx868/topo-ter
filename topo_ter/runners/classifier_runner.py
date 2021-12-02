import os.path

import torch

import torch.nn as nn

from tensorboardX import SummaryWriter

from topo_ter.datasets.data_provider import DataProvider
from topo_ter.models.backbone import Backbone
from topo_ter.models.classifier import Classifier
from topo_ter.runners.runner import Runner


class ClassifierRunner(Runner):
    def __init__(self, args):
        super(ClassifierRunner, self).__init__(args)
        # loss
        self.loss = nn.NLLLoss().to(self.output_dev)

    def load_dataset(self):
        self.dataset = DataProvider(
            data_path=self.args.data_path,
            dataset=self.args.dataset,
            stage='supervised'
        )
        data = self.dataset.next()
        self.x = data['x'].to(self.output_dev)
        self.y = data['y'].to(self.output_dev)
        self.edge_index = data['original_edge'].to(self.output_dev)
        self.train_mask = data['train_mask'].to(self.output_dev)
        self.test_mask = data['test_mask'].to(self.output_dev)

    def load_model(self):
        classifier = Classifier(
            in_channels=self.args.hidden_channels,
            num_classes=self.dataset.num_classes
        )
        self.model['train'] = classifier.to(self.output_dev)
        backbone = Backbone(
            in_channels=self.dataset.num_features,
            hidden_channels=self.args.hidden_channels,
            k=self.args.k
        )
        self.model['test'] = backbone.to(self.output_dev)

    def initialize_model(self):
        if self.args.backbone is not None:
            self.load_model_weights(self.model['test'], self.args.backbone)
        else:
            raise ValueError('Please appoint --backbone.')

        if self.args.classifier is not None:
            self.load_model_weights(self.model['train'], self.args.classifier)

    def run(self):
        # load embeddings
        self.features = self.model['test'](self.x, self.edge_index)
        # find epoch
        num_epochs = self._search_epoch()
        accuracies = []
        for t in range(50):
            self.load_model()
            self.load_optimizer()
            self.initialize_model()
            for epoch in range(num_epochs):
                self._train_classifier(epoch)
            acc = self._eval_classifier(t)
            accuracies.append(acc)
            self.print_log('Eval Round {}: Accuracy {:.2f}%'.format(
                t, acc.item() * 100.0
            ))
            if self.args.save_model:
                model_path = os.path.join(
                    self.args.save_dir, f'{self.args.dataset}_classifier_{t}.pt'
                )
                self.save_weights(self.model['train'], model_path)
        accuracies = torch.stack(accuracies)
        mean_acc = accuracies.mean()
        mean_std = accuracies.std()
        self.print_log('Mean accuracy: {:.2f}% +/- {:.2f}%'.format(
            mean_acc * 100.0, mean_std * 100.0
        ))

    def _search_epoch(self):
        # this part is borrowed from: https://github.com/zpeng27/GMI
        self.load_model()
        self.load_optimizer()
        self.initialize_model()
        epoch_flag = 0
        epoch_win = 0
        best_acc = 0
        for epoch in range(20000):
            self._train_classifier(epoch)
            if (epoch + 1) % 100 == 0:
                acc = self._eval_classifier(epoch)
                if acc >= best_acc:
                    epoch_flag = epoch + 1
                    best_acc = acc
                    epoch_win = 0
                else:
                    epoch_win += 1
                if epoch_win == 10:
                    break
        return epoch_flag

    def _train_classifier(self, epoch):
        self.model['train'].train()
        # forward
        pred = self.model['train'](self.features)
        loss = self.loss(pred[self.train_mask, :], self.y[self.train_mask])
        # backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('train/classifier_loss', loss.item(), epoch)
        return loss.item()

    def _eval_classifier(self, train_round):
        self.model['train'].eval()
        pred = self.model['train'](self.features)
        pred = pred[self.test_mask, :]
        gt = self.y[self.test_mask]
        loss = self.loss(pred, gt)

        pred = pred.max(dim=1)[1]
        correct = pred.eq(gt).double()
        acc = correct.sum() / gt.size(0)

        if self.args.use_tensorboard:
            with SummaryWriter(log_dir=self.tensorboard_path) as writer:
                writer.add_scalar('test/loss', loss.item(), train_round)
                writer.add_scalar('test/accuracy', acc.item(), train_round)
        return acc
