import os
import argparse

import torch
from torch.backends import cudnn

from model import build_model
from utils import set_random_seed
from __init__ import build_handler
import torch.multiprocessing
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
def parse_args():
    parser = argparse.ArgumentParser(description='base self framework')
    parser.add_argument('--config', default='./config/cfg_train.py', help='config file path')
   # parser.add_argument('--config', default='./config/cfg_test.py', help='config file path')
    parser.add_argument('--cuda', default='0', help='set cuda num')
    parser.add_argument('--cudnn_benchmark', default=True)
    parser.add_argument('--cudnn_deterministic', default=True)
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    from mmengine import Config
    cfg = Config.fromfile(args.config)

    if torch.cuda.is_available():
        cfg.cuda = args.cuda
    cfg.cudnn_benchmark = args.cudnn_benchmark
    cfg.cudnn_deterministic = args.cudnn_deterministic

    # set cudnn_benchmark & cudnn_deterministic
    cudnn.benchmark = cfg.cudnn_benchmark
    cudnn.deterministic = cfg.cudnn_deterministic

    # set seed
    set_random_seed(cfg.seed)

    model = build_model(cfg.model['type'], **cfg.model['kwargs'])

    handler = build_handler(cfg.phase, cfg.model['type'])

    handler(model, cfg)

if __name__ == '__main__':
    main()


