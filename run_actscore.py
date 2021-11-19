from __future__ import division

import os
import json
import argparse
import numpy as np

import torch

from utils.args import actscore_parser
from tasks.actscore import ActSCoRe
from utils.mypath import mypath

from itertools import product

import warnings
warnings.filterwarnings(action='ignore')


def my_product(inp):
    return (dict(zip(inp.keys(), values)) for values in product(*inp.values()))


def main(args: argparse, game_name: str = None, exp_num: int = None):

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

    actscore = ActSCoRe(args, results_dir)
    actscore.run_actscore()


if __name__ == '__main__':
    parser = actscore_parser()
    args = parser.parse_args()

    # game_list = ['alien', 'amidar', 'assault', 'asterix', 'bank_heist',
    #              'battle_zone', 'boxing', 'breakout', 'chopper_command',
    #              'crazy_climber', 'demon_attack', 'freeway', 'frostbite',
    #              'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull',
    #              'kung_fu_master', 'ms_pacman', 'pong', 'private_eye',
    #              'qbert', 'road_runner', 'seaquest', 'up_n_down']

    game_list = ['kangaroo']
    
    experiments = [1840, 9178, 2885, 11697, 4690,
                   6877, 2436, 7749, 2584, 396]

    if args.ssl_option == 'none':
        # Baseline
        for i, exp_seed in enumerate(experiments):
            args.seed = int(exp_seed)
            args.stcl_option = 'stdim'
            args.ucb_option = False

            for game_name in game_list:
                main(args, game_name, exp_num=int(i+1))

    elif args.ssl_option == 'actscore':

        # Proposed Method (quanile)
        if args.threshold_option == 'quantile':
            for i, exp_seed in enumerate(experiments):
                args.seed = int(exp_seed)
                args.stcl_option = 'stdim'  # Fix
                args.ucb_option = True  # Fix
                args.num_threshold = 0.5  # Change

                for game_name in game_list:
                    main(args, game_name, exp_num=int(i+1))

        # Proposed Method (topk)
        elif args.threshold_option == 'topk':
            for i, exp_seed in enumerate(experiments):
                args.seed = int(exp_seed)
                args.stcl_option = 'stdim'  # Fix
                args.ucb_option = True  # Fix
                args.num_topk = 16  # Change

                for game_name in game_list:
                    main(args, game_name, exp_num=int(i+1))

        else:
            # Proposed Method (Original SCL)
            for i, exp_seed in enumerate(experiments):
                args.seed = int(exp_seed)
                args.stcl_option = 'stdim'  # Fix
                args.ucb_option = True  # Fix

                for game_name in game_list:
                    main(args, game_name, exp_num=int(i+1))
