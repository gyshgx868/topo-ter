import torch

import torch.nn as nn

from torch_geometric.nn import SGConv


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels=512, k=1):
        super(Encoder, self).__init__()
        self.conv0 = SGConv(in_channels, out_channels, K=k)
        self.relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x, edge_index):
        y = self.conv0(x, edge_index)
        y = self.relu(y)
        return y


class Backbone(nn.Module):
    def __init__(self, in_channels, hidden_channels=512, k=1,
                 fusion_scheme='minus'):
        super(Backbone, self).__init__()
        self.fusion_scheme = fusion_scheme
        self.encoder = Encoder(
            in_channels=in_channels, out_channels=hidden_channels, k=k
        )
        if self.fusion_scheme == 'concat':
            hidden_channels *= 2
        self.decoder = nn.Linear(hidden_channels, 4)

    def forward(self, *args):
        if len(args) == 4:
            x = args[0]
            edge_index1, edge_index2 = args[1], args[2]
            modified = args[3]
            x1 = self.encoder(x, edge_index1)
            x2 = self.encoder(x, edge_index2)

            if self.fusion_scheme == 'minus':
                x = x2 - x1
            elif self.fusion_scheme == 'add':
                x = x1 + x2
            elif self.fusion_scheme == 'concat':
                x = torch.cat((x1, x2), dim=-1)
            else:
                raise ValueError('Unknown fusion scheme.')

            left_nodes = x[modified[0]]
            right_nodes = x[modified[1]]
            import torch.nn.functional as F
            x = F.softmax(-(left_nodes - right_nodes)**2, dim=-1)
            t_hat = self.decoder(x)
            return t_hat
        elif len(args) == 2:
            x, edge_index = args[0], args[1]
            features = self.encoder(x, edge_index)
            return features.detach()
        else:
            raise ValueError('Invalid number of arguments.')
