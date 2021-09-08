import os
import argparse


def mypath(args: argparse, exp_num: int = None):

    result_path = f'./{args.stcl_option}_{args.ssl_option}_{args.pos_threshold}_results/Experiment_{exp_num}'

    return result_path

