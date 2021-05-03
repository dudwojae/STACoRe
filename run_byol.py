from __future__ import division

import os
import json
import numpy as np

import torch

from utils.args import byol_argparser
from tasks.byol import BYOL_rainbow
from utils.mypath import result_path


def main(args, game_name):
    args.game = game_name
    xid = f'byol-{args.game}-{str(args.seed)}'
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
    torch.manual_seed(np.random.randint(1, 10000))
    torch.cuda.manual_seed(np.random.randint(1, 10000))
    torch.backends.cudnn.enabled = args.enable_cudnn

    byol_rainbow = BYOL_rainbow(args, results_dir)
    byol_rainbow.run_byol_rainbow()


if __name__ == '__main__':
    parser = byol_argparser()
    args = parser.parse_args()

    game_list = ['alien', 'amidar', 'assault', 'asterix', 'bank_heist',
                 'battle_zone', 'boxing', 'breakout', 'chopper_command',
                 'crazy_climber', 'demon_attack', 'freeway', 'frostbite',
                 'gopher', 'hero', 'jamesbond', 'kangaroo', 'krull',
                 'kung_fu_master', 'ms_pacman', 'pong', 'private_eye',
                 'qbert', 'road_runner', 'seaquest', 'up_n_down']

    for game_name in game_list:
        main(args, game_name)
