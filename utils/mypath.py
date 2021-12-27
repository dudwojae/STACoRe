import os
import argparse


def mypath(args: argparse, exp_num: int = None):

    if args.threshold_option == 'quantile':

        if args.ucb_option:
            result_path = f'./{args.stcl_option}_{args.ssl_option}_' \
                          f'{args.num_threshold}_results/Experiment_{exp_num}'

        else:
            result_path = f'./{args.stcl_option}_{args.ssl_option}_' \
                          f'{args.num_threshold}_UCBNone_results/Experiment_{exp_num}'

    elif args.threshold_option == 'topk':
        result_path = f'./{args.stcl_option}_{args.ssl_option}_' \
                      f'{args.num_topk}_results/Experiment_{exp_num}'

    else:
        result_path = f'./{args.stcl_option}_{args.ssl_option}_' \
                      f'none_results/Experiment_{exp_num}'

    return result_path

