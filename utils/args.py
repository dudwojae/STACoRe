import atari_py
import argparse


def stcl_parser():
    parser = argparse.ArgumentParser(description='Self-supervised learning coupled with efficient Rainbow')

    # environment option (Don't Change)
    parser.add_argument('--id', type=str, default='default',
                        help='Experiment ID')
    parser.add_argument('--max_episode_length', type=int,
                        default=int(108e3), metavar='LENGTH',
                        help='Max episode length in game frames (0 to disable)')
    parser.add_argument('--game', type=str, default='ms_pacman',
                        choices=atari_py.list_games(),
                        help='Atari game')
    parser.add_argument('--history_length', type=int,
                        default=4, metavar='T',
                        help='Number of consecutive states processed')
    parser.add_argument('--atoms', type=int,
                        default=51, metavar='C',
                        help='Discretized size of value distribution')

    # Rainbow parameter (Don't Change)
    # FIXME: architecture data_effieicnt -> canonical (Same as SPR)
    parser.add_argument('--architecture', type=str,
                        default='canonical', metavar='ARCH',
                        choices=['canonical', 'data_efficient'],
                        help='Network architecture')
    parser.add_argument('--V_min', type=float, default=-10, metavar='V',
                        help='Minimum of value distribution support')
    parser.add_argument('--V_max', type=float, default=10, metavar='V',
                        help='Maximum of value distribution support')

    # state option (Don't Change)
    parser.add_argument('--resize', type=int, default=84,
                        help='Resize state information')

    # Noisy network parameter (Don't Change)
    parser.add_argument('--noisy_std', type=float, default=0.5, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='SIZE',
                        help='Network hidden size')

    # Replay memory parameter (Don't Change)
    parser.add_argument('--priority_exponent', type=float, default=0.5, metavar='ω',
                        help='Prioritized experience replay exponent (originally denoted α)')
    parser.add_argument('--priority_weight', type=float, default=1, metavar='β',
                        help='Initial prioritised experience replay importance sampling weight')
    parser.add_argument('--memory', help='Path to save/load the memory from')
    parser.add_argument('--disable_bzip_memory', action='store_true',
                        help='Don\'t zip the memory file. '
                             'Not recommended (zipping is a bit slower and much, much smaller)')
    parser.add_argument('--memory_capacity', type=int, default=int(1e5),
                        metavar='CAPACITY',
                        help='Experience replay memory capacity')

    # Training hyperparamters (Don't Change)
    # FIXME: multi_step 20 -> 10 (Same as SPR)
    parser.add_argument('--multi_step', type=int, default=10, metavar='n',
                        help='Number of steps for multi-step return')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Reward discount factor')
    parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE',
                        help='Batch size')
    parser.add_argument('--model', type=str, metavar='PARAMS',
                        help='Pretrained model (state dict)')
    # FIXME: learn_start 1600 -> 2000 (Same as SPR)
    parser.add_argument('--learn_start', type=int, default=int(2e3), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--T_max', type=int, default=int(1e5), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--replay_frequency', type=int, default=1, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--reward_clip', type=int, default=1, metavar='VALUE',
                        help='Reward clipping (0 to disable)')
    # FIXME: target_update 2000 -> 1 (Same as SPR)
    parser.add_argument('--target_update', type=int, default=1, metavar='τ',
                        help='Number of steps after which to update target network')
    # FIXME: Find Optimal Lambda
    parser.add_argument('--lambda_coef', type=float, default=1.,
                        help='Weighted contrastive loss coefficient')

    # optimizer parameters (Don't Change)
    parser.add_argument('--clip_value', type=float, default=10, metavar='NORM',
                        help='Max L2 norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε',
                        help='Adam epsilon')

    # Evaluate parameter (Don't Change)
    parser.add_argument('--evaluate', type=bool, default=False,
                        help='Evaluate only')
    parser.add_argument('--evaluation_interval', type=int, default=10000, metavar='STEPS',
                        help='Number of training steps between evaluations')
    parser.add_argument('--evaluation_episodes', type=int, default=10, metavar='N',
                        help='Number of evaluation episodes to average over')
    parser.add_argument('--evaluation_size', type=int, default=500, metavar='N',
                        help='Number of transitions to use for validating Q')
    parser.add_argument('--render', type=bool, default=False,
                        help='Display screen (testing only)')
    parser.add_argument('--checkpoint_interval', default=5000,
                        help='How often to checkpoint the model, defaults to 0 (never checkpoint)')

    # Self-Supervised Learning & ST-DIM parameter
    parser.add_argument('--projection_size', type=int, default=256, metavar='SIZE',
                        help='Network hidden size')
    parser.add_argument('--local_depth', type=int, default=64, metavar='SIZE',
                        help='Feature map depth size')
    parser.add_argument('--temperature', type=float, default=0.5,
                        help='Logit scaling factor (SimCLR)')

    # Supervised Contrastive Learning parameter
    parser.add_argument('--scl_temperature', type=float, default=0.1,
                        help='Temperature for loss function (SupCon)')
    parser.add_argument('--base_scl_temperature', type=float, default=0.1,
                        help='SupCon temperature scaling factor (SupCon)')
    parser.add_argument('--pos_candidate', type=int, default=32,
                        choices=[32, 16, 8],
                        help='Number of false negative or positive candidate based on euclidean distance')

    # Experiment Option (UCB, STDIM, SSL)
    parser.add_argument('--ucb_option', type=bool,
                        default=True, help='UCB Multi-Armed Bandit Switch')
    parser.add_argument('--stcl_option', type=str,
                        default='stdim', metavar='ARCH',
                        choices=['stdim', 'none'],
                        help='SpatioTemporal Contrastive Learning Method Switch')
    parser.add_argument('--ssl_option', type=str,
                        default='simclr', metavar='ARCH',
                        choices=['simclr' 'supcon', 'none'],
                        help='Self-Supervised/Supervised Contrastive Learning Method Switch')

    # Upper Confidence Bound Multi-Armed Bandit Problem parameter
    parser.add_argument('--ucb_exploration_coef', type=float, default=0.5,
                        help='UCB Exploration coefficient')
    parser.add_argument('--ucb_window_length', type=int, default=10,
                        help='Sliding Window Average of the Past K mean returns')
    parser.add_argument('--random_choice_step', type=int, default=0,
                        help='N-step random choice for exploring UCB')

    # cuda and seed
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Ables CUDA training (default: cuda:0)')
    parser.add_argument('--enable_cudnn', action='store_true',
                        help='Enable cuDNN (faster but nondeterministic)')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed (default: auto)')

    return parser
