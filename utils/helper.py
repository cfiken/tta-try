import numpy as np
from argparse import ArgumentParser
import yaml


def assert_shape(target: np.ndarray, shape):
    if isinstance(shape, list):
        shape = np.array(shape)
    
    assert isinstance(target, np.ndarray), "target is not ndarray"
    assert isinstance(shape, np.ndarray), "shape is not ndarray"
    
    target_shape = target.shape
    assert len(target_shape) == len(shape), 'dimension is not equal'
    for (i, s) in enumerate(shape):
        assert target_shape[i] == s, "size of target[{}] is not {}".format(i, s)


def get_args():
    argparser = ArgumentParser()
    argparser.add_argument('-c', '--config', 
                           default='None',
                           help='The Configuration File')
    args = argparser.parse_args()
    return args


def load_config(config_path: str):
    f = open(config_path, 'r+')
    config = yaml.load(f)
    return config
