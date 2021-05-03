from __future__ import division

import os
import json
import numpy as np

import torch

from utils.args import Curl_Rainbow_parser
from tasks.curl import Curl_Rainbow
from utils.mypath import result_path


def main(args):
    xid = f'curl-{args.game}-{str(args.seed)}'
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

    curl_rainbow = Curl_Rainbow(args, results_dir)
    curl_rainbow.run_curl_rainbow()


if __name__ == '__main__':
    parser = Curl_Rainbow_parser()
    args = parser.parse_args()

    main(args)
