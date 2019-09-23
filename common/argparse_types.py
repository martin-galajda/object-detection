import argparse


def str2bool(v: str):
    """
    Defines how to parse string argument to boolean value provided 
    in CLI programs.
    """
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
