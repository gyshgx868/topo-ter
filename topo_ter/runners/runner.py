import collections
import json
import os
import random
import time
import torch
import yaml

import numpy as np

from abc import abstractmethod


class Runner:
    def __init__(self, args):
        # register variables
        self.backbone_path = None
        self.classifier_path = None
        self.tensorboard_path = None

        self.dataset = None
        self.model = {'train': None, 'test': None}
        self.optimizer = None

        self.cur_time = 0
        self.epoch = 0

        # check arguments
        self.args = self.check_args(args)
        # print args
        args_json = json.dumps(vars(self.args), sort_keys=True, indent=2)
        self.print_log(args_json, print_time=False)

        # random seed
        if self.args.use_seed:
            random.seed(args.seed)
            np.random.seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.manual_seed(args.seed)
            if self.args.use_cuda:
                torch.cuda.manual_seed(args.seed)
        # devices
        if self.args.use_cuda:
            self.output_dev = args.device
        else:
            self.output_dev = 'cpu'
        # model
        self.load_dataset()
        self.load_model()
        self.load_optimizer()
        self.initialize_model()

    def check_args(self, args):
        self.tensorboard_path = os.path.join(args.save_dir, 'tensorboard')
        if not os.path.exists(self.tensorboard_path):
            os.makedirs(self.tensorboard_path)

        args.use_cuda = args.use_cuda and torch.cuda.is_available()
        args.num_epochs = max(1, args.num_epochs)

        # save configuration file
        config_file = os.path.join(args.save_dir, 'config.yaml')
        args_dict = vars(args)
        with open(config_file, 'w') as f:
            yaml.dump(args_dict, f)
        return args

    @abstractmethod
    def load_dataset(self):
        pass

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def initialize_model(self):
        pass

    def load_optimizer(self):
        if self.model['train'] is None:
            return
        if 'sgd' in self.args.optimizer.lower():
            self.optimizer = torch.optim.SGD(
                self.model['train'].parameters(),
                lr=self.args.lr,
                momentum=self.args.momentum,
                weight_decay=0
            )
        elif 'adam' in self.args.optimizer.lower():
            self.optimizer = torch.optim.Adam(
                self.model['train'].parameters(),
                lr=self.args.lr,
                weight_decay=0
            )
        else:
            raise ValueError('Unsupported optimizer.')

    @abstractmethod
    def run(self):
        pass

    def load_model_weights(self, model, weights_file):
        # self.print_log(f'Loading model weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load model weights
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.to(self.output_dev))
            for k, v in check_points.items()
        ])
        self._try_load_weights(model, model_weights)

    def load_optimizer_weights(self, optimizer, weights_file):
        self.print_log(f'Loading optimizer weights from {weights_file}...')
        check_points = torch.load(
            weights_file, map_location=lambda storage, loc: storage
        )
        # load optimizer configuration
        optim_weights = check_points['optimizer']
        self._try_load_weights(optimizer, optim_weights)

    def _try_load_weights(self, model, weights):
        try:
            model.load_state_dict(weights)
        except:
            state = model.state_dict()
            diff = list(set(state.keys()).difference(set(weights.keys())))
            self.print_log('Can not find these weights:')
            for d in diff:
                self.print_log(d)
            state.update(weights)
            model.load_state_dict(state)

    def save_weights(self, model, save_path):
        model_weights = collections.OrderedDict([
            (k.split('module.')[-1], v.cpu())
            for k, v in model.state_dict().items()
        ])
        torch.save(model_weights, save_path)

    def record_time(self):
        self.cur_time = time.time()
        return self.cur_time

    def tick(self):
        split_time = time.time() - self.cur_time
        self.record_time()
        return split_time

    def print_log(self, msg, print_time=True):
        # pass
        if print_time:
            localtime = time.strftime("%Y/%m/%d %H:%M:%S", time.localtime())
            msg = "[" + localtime + '] ' + msg
        print(msg)
        if self.args.print_log:
            with open(os.path.join(self.args.save_dir, 'log.txt'), 'a') as f:
                print(msg, file=f)
