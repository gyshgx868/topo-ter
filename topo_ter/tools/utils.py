import argparse


def import_class(name):
    try:
        components = name.split('.')
        module = __import__(components[0])
        for c in components[1:]:
            module = getattr(module, c)
    except AttributeError:
        module = None
    return module


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def get_total_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total, 'Trainable': trainable}
