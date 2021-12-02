import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, in_channels=512, num_classes=3):
        super(Classifier, self).__init__()
        self.classifier = nn.Linear(in_channels, num_classes, bias=True)
        self.softmax = nn.LogSoftmax(dim=1)

        nn.init.xavier_uniform_(self.classifier.weight.data)
        self.classifier.bias.data.fill_(0.0)

    def forward(self, x):
        y = self.classifier(x)
        y = self.softmax(y)
        return y


def main():
    from topo_ter.tools.utils import get_total_parameters
    layer = nn.Linear(in_features=2, out_features=4, bias=True)
    print(get_total_parameters(layer))


if __name__ == '__main__':
    main()
