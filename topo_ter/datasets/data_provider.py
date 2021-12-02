import torch

import numpy as np
import torch_geometric.transforms as T

from torch_geometric.datasets import Planetoid


class DataProvider:
    def __init__(self, data_path, dataset='cora', perturbation_rate=0.7,
                 stage='unsupervised'):
        self.perturbation_rate = perturbation_rate
        self.stage = stage
        planetoid = Planetoid(data_path, dataset, transform=T.TargetIndegree())
        data = planetoid.data

        self.features = data.x.numpy()
        self.labels = data.y.numpy()
        self.edge_index = data.edge_index.numpy()

        self.train_mask = data.train_mask.numpy()
        self.test_mask = data.test_mask.numpy()
        self.val_mask =  data.val_mask.numpy()

        self.train_mask = np.where(self.train_mask == 1)[0]
        self.test_mask = np.where(self.test_mask == 1)[0]
        self.val_mask = np.where(self.val_mask == 1)[0]

        self.num_features = self.features.shape[1]
        self.num_vertices = self.features.shape[0]
        self.num_classes = np.max(self.labels) + 1

        self.unique_edges = set()
        self.to_modify = set()
        for i in range(self.edge_index.shape[1]):
            left = self.edge_index[0, i]
            right = self.edge_index[1, i]
            self.unique_edges.add((min(left, right), max(left, right)))
            self.to_modify.add((left, right))
        self.adj_list = dict()
        for i in range(self.num_vertices):
            self.adj_list.update({i: set()})
        for (left, right) in self.unique_edges:
            self.adj_list[left].add(right)

        self._to_tensor()

    def _to_tensor(self):
        self.labels = torch.LongTensor(self.labels)
        self.features = torch.FloatTensor(self.features)
        self.edge_index = torch.LongTensor(self.edge_index)
        self.train_mask = torch.LongTensor(self.train_mask)
        self.test_mask = torch.LongTensor(self.test_mask)
        self.val_mask = torch.LongTensor(self.val_mask)

    def _random_perturb(self):
        to_add = set()
        to_discard = set()
        to_remain = set()
        to_remain_empty = set()

        for (left, right) in self.unique_edges:
            new_right = np.random.randint(0, self.num_vertices)
            t = (min(left, new_right), max(left, new_right))
            while t in self.unique_edges or t in to_add or t in to_discard or \
                    t in to_remain or t in to_remain_empty or left == new_right:
                new_right = np.random.randint(0, self.num_vertices)
                t = (min(left, new_right), max(left, new_right))
            if np.random.rand() > self.perturbation_rate:
                to_remain.add((left, right))
                to_remain_empty.add(t)
            else:
                to_add.add(t)
                to_discard.add((left, right))

        sampled = []
        edge_types = []
        perturbed_adj = self.to_modify.copy()
        for (left, right) in to_add:
            assert (left, right) not in to_discard
            assert (left, right) not in to_remain
            assert (left, right) not in to_remain_empty
            perturbed_adj.add((left, right))
            perturbed_adj.add((right, left))
            sampled.append((min(left, right), max(left, right)))
            edge_types.append(3)
        for (left, right) in to_discard:
            assert (left, right) not in to_add
            assert (left, right) not in to_remain
            assert (left, right) not in to_remain_empty
            perturbed_adj.remove((left, right))
            perturbed_adj.remove((right, left))
            sampled.append((min(left, right), max(left, right)))
            edge_types.append(2)
        for (left, right) in to_remain:
            assert (left, right) not in to_add
            assert (left, right) not in to_discard
            assert (left, right) not in to_remain_empty
            sampled.append((min(left, right), max(left, right)))
            edge_types.append(1)
        for (left, right) in to_remain_empty:
            assert (left, right) not in to_add
            assert (left, right) not in to_discard
            assert (left, right) not in to_remain
            sampled.append((min(left, right), max(left, right)))
            edge_types.append(0)

        sampled = np.array(sampled, dtype=int).T
        edge_types = np.array(edge_types)
        perturbed_adj = list(perturbed_adj)
        perturbed_adj = np.array(perturbed_adj).T
        return perturbed_adj, sampled, edge_types

    def next(self):
        if self.stage == 'unsupervised':
            perturbed_edge, sampled, edge_types = self._random_perturb()
            perturbed_edge = torch.LongTensor(perturbed_edge)
            sampled = torch.LongTensor(sampled)
            edge_types = torch.LongTensor(edge_types)
        else:
            perturbed_edge, sampled, edge_types = None, None, None

        return {
            'x': self.features,
            'y': self.labels,
            'original_edge': self.edge_index,
            'perturbed_edge': perturbed_edge,
            'sampled_edge': sampled,
            'perturbation_type': edge_types,
            'train_mask': self.train_mask,
            'val_mask': self.val_mask,
            'test_mask': self.test_mask
        }


def main():
    cora_dataset = DataProvider(data_path='../../data', dataset='cora')
    print('Cora Dataset:')
    print('  num_features:', cora_dataset.num_features)
    print('  num_vertices:', cora_dataset.num_vertices)
    print('  num_classes:', cora_dataset.num_classes)

    data = cora_dataset.next()
    x = data['x']
    original_edge = data['original_edge']
    perturbed_edge = data['perturbed_edge']
    sampled = data['sampled_edge']
    edge_types = data['perturbation_type']
    print(x.size())
    print(original_edge.size(), perturbed_edge.size())
    print(sampled.size(), edge_types.size())
    print(sampled)

    citeseer_dataset = DataProvider(data_path='../../data', dataset='citeseer')
    print('Citeseer Dataset:')
    print('  num_features:', citeseer_dataset.num_features)
    print('  num_vertices:', citeseer_dataset.num_vertices)
    print('  num_classes:', citeseer_dataset.num_classes)

    pubmed_dataset = DataProvider(data_path='../../data', dataset='pubmed')
    print('Pubmed Dataset:')
    print('  num_features:', pubmed_dataset.num_features)
    print('  num_vertices:', pubmed_dataset.num_vertices)
    print('  num_classes:', pubmed_dataset.num_classes)


if __name__ == '__main__':
    main()
