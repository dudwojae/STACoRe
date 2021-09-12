import os
import argparse


def mypath(args: argparse, exp_num: int = None):

    if args.threshold_option == 'quantile':
        result_path = f'./{args.stcl_option}_{args.ssl_option}_{args.num_threshold}_results/Experiment_{exp_num}'

    elif args.threshold_option == 'topk':
        result_path = f'./{args.stcl_option}_{args.ssl_option}_{args.num_topk}_results/Experiment_{exp_num}'

    else:
        result_path = f'./{args.stcl_option}_{args.ssl_option}_none_results/Experiment_{exp_num}'

    return result_path

