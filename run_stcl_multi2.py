from __future__ import division

import os
import json
import argparse
import numpy as np

import torch

from utils.args import stcl_parser
from tasks.stcl import STCL_Rainbow
from utils.mypath import mypath

from itertools import product

import warnings
warnings.filterwarnings(action='ignore')

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import multiprocessing as mp

def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def main(args: argparse, game_name: str = None, exp_num: int = None):
    try:
        # Define Save Path
        result_path = mypath(args, exp_num=exp_num)

        args.game = game_name
        xid = f'{args.game}-{str(args.seed)}'
        args.id = xid

        print(' ' * 26 + 'Options')
        for k, v in vars(args).items():
            print(' ' * 26 + k + ': ' + str(v))

        results_dir = os.path.join(result_path, args.id)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        # Save parameter argument dictionary
        with open(os.path.join(results_dir, 'arg_parser.json'), 'w') as f:
            json.dump(args.__dict__, f, indent=2)

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.enabled = args.enable_cudnn

        stcl_rainbow = STCL_Rainbow(args, results_dir)
    except Exception as e:
        print(e)
    else:
        stcl_rainbow.run_stcl_rainbow()


def is_exist(root='stdim_supcon_4_results/'):
    dirs = []
    for d in os.listdir(root):
        dirs += os.listdir(root + d)
    return dirs


def main_par(experiments):
    pool = mp.Pool(5)
    parser = stcl_parser()
    args = parser.parse_args()

    game_list = ['alien', 'amidar', 'assault', 'asterix', 'bank_heist',
                 'battle_zone', 'boxing', 'breakout', 'chopper_command',
                 'crazy_climber', 'demon_attack', 'freeway', 'frostbite',
                 'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull',
                 'kung_fu_master', 'ms_pacman', 'pong', 'private_eye',
                 'qbert', 'road_runner', 'seaquest', 'up_n_down']

    jobs = []
    for i, exp_seed in enumerate(experiments):
        if i >= 5:
            args.seed = int(exp_seed)
            args.stcl_option = 'stdim'  # Fix
            args.ucb_option = True  # Fix
            args.ssl_option = 'supcon'  # Fix
            args.num_topk = 4  # Change
            for game_name in game_list:
                dirs = is_exist()
                key = game_name + '-' + str(exp_seed)
                if key not in dirs:
                    jobs.append(pool.apply_async(main, (args, game_name, int(i+1))))
                else:
                    print(key, 'already exist')

    for j in jobs:
        j.get()

    pool.close()


if __name__ == '__main__':
    experiments = [1840, 9178, 2885, 11697, 4690,
                   6877, 2436, 7749, 2584, 396]

    main_par(experiments)

