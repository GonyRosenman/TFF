from data_preprocess_and_load.datasets import *
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from datetime import datetime
import argparse
import os

def datestamp():
    time = datetime.now().strftime("%d_%m___%H_%M_%S")
    return time

def reproducibility(**kwargs):
    seed = kwargs.get('seed')
    cuda = kwargs.get('cuda')
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

def sort_args(phase, args):
    phase_specific_args = {}
    for name, value in args.items():
        if not 'phase' in name:
            phase_specific_args[name] = value
        elif 'phase' + phase in name:
            phase_specific_args[name.replace('_phase' + phase, '')] = value
    return phase_specific_args
