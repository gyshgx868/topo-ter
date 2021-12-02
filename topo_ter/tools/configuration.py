import argparse
import json

from topo_ter.tools.utils import str2bool


def get_parser():
    parser = argparse.ArgumentParser(
        description='Graph Transformation Equivariant Representations Network'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='path to the configuration file'
    )

    # model hyper-parameters
    parser.add_argument(
        '--use-seed',
        type=str2bool,
        default='false',
        help='whether to use random seed'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=6747,
        help='random seed for PyTorch and NumPy'
    )

    # runner
    parser.add_argument(
        '--use-cuda',
        type=str2bool,
        default='false',
        help='whether to use GPUs'
    )
    parser.add_argument(
        '--device',
        type=int,
        default=0,
        help='the indices of GPU for training or testing'
    )
    parser.add_argument(
        '--phase',
        type=str,
        default='backbone',
        help='it must be \'backbone\' or \'classifier\''
    )
    parser.add_argument(
        '--perturbation-rate',
        type=float,
        default=0.5,
        help='edge perturbation rate'
    )
    parser.add_argument(
        '--hidden-channels',
        type=int,
        default=512,
        help='channels of the hidden embeddings'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=2,
        help='number of hops'
    )
    parser.add_argument(
        '--num-epochs',
        type=int,
        default=200,
        help='maximum number of training epochs'
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=20,
    )
    parser.add_argument(
        '--backbone',
        default=None,
        help='the weights for backbone initialization'
    )
    parser.add_argument(
        '--classifier',
        default=None,
        help='the weights for classifier initialization'
    )
    parser.add_argument(
        '--eval-model',
        type=str2bool,
        default='true',
        help='if true, the model will be evaluated during training'
    )
    parser.add_argument(
        '--eval-interval',
        type=int,
        default=100,
        help='the interval for evaluating models (#epoch)'
    )

    # dataset
    parser.add_argument(
        '--data-path',
        type=str,
        default='./data',
        help='dataset path'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='cora',
        help='dataset name'
    )

    # optimizer
    parser.add_argument(
        '--optimizer',
        type=str,
        default='adam',
        help='optimizer to use'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=0.1,
        help='initial learning rate (default: 0.1)'
    )
    parser.add_argument(
        '--momentum',
        type=float,
        default=0.9,
        help='SGD momentum (default: 0.9)'
    )

    # logging
    parser.add_argument(
        '--save-dir',
        type=str,
        default='./results',
        help='path to save results'
    )
    parser.add_argument(
        '--save-model',
        type=str2bool,
        default='true',
        help='whether to save model'
    )
    parser.add_argument(
        '--print-log',
        type=str2bool,
        default='true',
        help='whether to print logs'
    )
    parser.add_argument(
        '--use-tensorboard',
        type=str2bool,
        default='false',
        help='whether to use TensorBoard to visualize results'
    )

    return parser


def main():
    p = get_parser()
    js = json.dumps(vars(p), indent=2)
    print(js)


if __name__ == '__main__':
    main()
