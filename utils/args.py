import atari_py
import argparse
import numpy as np


def Curl_Rainbow_parser():
    parser = argparse.ArgumentParser(description='Curl coupled with efficient Rainbow')

    # environment option
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

    # Rainbow parameter
    parser.add_argument('--architecture', type=str,
                        default='data_efficient', metavar='ARCH',
                        choices=['canonical', 'data_efficient'],
                        help='Network architecture')
    parser.add_argument('--V_min', type=float, default=-10, metavar='V',
                        help='Minimum of value distribution support')
    parser.add_argument('--V_max', type=float, default=10, metavar='V',
                        help='Maximum of value distribution support')

    # state option
    parser.add_argument('--resize', type=int, default=84,
                        help='Resize state information')

    # Noisy network parameter
    parser.add_argument('--noisy_std', type=float, default=0.1, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='SIZE',
                        help='Network hidden size')

    # Replay memory parameter
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

    # Training hyperparamters
    parser.add_argument('--multi_step', type=int, default=20, metavar='n',
                        help='Number of steps for multi-step return')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Reward discount factor')
    parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE',
                        help='Batch size')
    parser.add_argument('--model', type=str, metavar='PARAMS',
                        help='Pretrained model (state dict)')
    parser.add_argument('--learn_start', type=int, default=int(1600), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--T_max', type=int, default=int(1e5), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--replay_frequency', type=int, default=1, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--reward_clip', type=int, default=1, metavar='VALUE',
                        help='Reward clipping (0 to disable)')
    parser.add_argument('--target_update', type=int, default=int(2e3), metavar='τ',
                        help='Number of steps after which to update target network')

    # optimizer parameters
    parser.add_argument('--clip_value', type=float, default=10, metavar='NORM',
                        help='Max L2 norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε',
                        help='Adam epsilon')

    # Evaluate parameter
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

    # cuda and seed
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Ables CUDA training (default: cuda:0)')
    parser.add_argument('--enable_cudnn', action='store_true',
                        help='Enable cuDNN (faster but nondeterministic)')
    seed = np.random.randint(12345)
    parser.add_argument('--seed', type=int, default=seed,
                        help='Random seed')

    return parser


def byol_argparser():
    parser = argparse.ArgumentParser(description='Byol coupled with efficient Rainbow')

    # environment option
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

    # Rainbow parameter
    parser.add_argument('--architecture', type=str,
                        default='data_efficient', metavar='ARCH',
                        choices=['canonical', 'data_efficient'],
                        help='Network architecture')
    parser.add_argument('--V_min', type=float, default=-10, metavar='V',
                        help='Minimum of value distribution support')
    parser.add_argument('--V_max', type=float, default=10, metavar='V',
                        help='Maximum of value distribution support')

    # state option
    parser.add_argument('--resize', type=int, default=84,
                        help='Resize state information')

    # Noisy network parameter
    # FIXME 0.1 -> 0.5
    parser.add_argument('--noisy_std', type=float, default=0.1, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='SIZE',
                        help='Network hidden size')

    # BYOL mlp parameter
    parser.add_argument('--byol_hidden_size', type=int, default=512, metavar='SIZE',
                        help='Network hidden size')
    parser.add_argument('--projection_size', type=int, default=128, metavar='SIZE',
                        help='Network hidden size')
    parser.add_argument('--momentum', type=float, default=0.001,
                        help='Momentum rate to BYOL target network update ')

    # Replay memory parameter
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

    # Training hyperparamters
    parser.add_argument('--multi_step', type=int, default=20, metavar='n',
                        help='Number of steps for multi-step return')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Reward discount factor')
    parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE',
                        help='Batch size')
    parser.add_argument('--model', type=str, metavar='PARAMS',
                        help='Pretrained model (state dict)')
    parser.add_argument('--learn_start', type=int, default=int(1600), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--T_max', type=int, default=int(1e5), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--replay_frequency', type=int, default=1, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--reward_clip', type=int, default=1, metavar='VALUE',
                        help='Reward clipping (0 to disable)')
    parser.add_argument('--target_update', type=int, default=int(2e3), metavar='τ',
                        help='Number of steps after which to update target network')

    # optimizer parameters
    parser.add_argument('--clip_value', type=float, default=10, metavar='NORM',
                        help='Max L2 norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε',
                        help='Adam epsilon')

    # Evaluate parameter
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

    # cuda and seed
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Ables CUDA training (default: cuda:0)')
    parser.add_argument('--enable_cudnn', action='store_true',
                        help='Enable cuDNN (faster but nondeterministic)')
    seed = np.random.randint(12345)
    parser.add_argument('--seed', type=int, default=seed,
                        help='Random seed')

    return parser


def simclr_parser():
    parser = argparse.ArgumentParser(description='SimCLR coupled with efficient Rainbow')

    # environment option
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

    # Rainbow parameter
    parser.add_argument('--architecture', type=str,
                        default='data_efficient', metavar='ARCH',
                        choices=['canonical', 'data_efficient'],
                        help='Network architecture')
    parser.add_argument('--V_min', type=float, default=-10, metavar='V',
                        help='Minimum of value distribution support')
    parser.add_argument('--V_max', type=float, default=10, metavar='V',
                        help='Maximum of value distribution support')

    # state option
    parser.add_argument('--resize', type=int, default=84,
                        help='Resize state information')

    # Noisy network parameter
    # FIXME 0.1 -> 0.5
    parser.add_argument('--noisy_std', type=float, default=0.5, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='SIZE',
                        help='Network hidden size')

    # SimCLR parameter
    parser.add_argument('--projection_size', type=int, default=128, metavar='SIZE',
                        help='Network hidden size')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Logit scaling factor')

    # Replay memory parameter
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
    
    # Training hyperparamters
    parser.add_argument('--multi_step', type=int, default=20, metavar='n',
                        help='Number of steps for multi-step return')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Reward discount factor')
    parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE',
                        help='Batch size')
    parser.add_argument('--model', type=str, metavar='PARAMS',
                        help='Pretrained model (state dict)')
    parser.add_argument('--learn_start', type=int, default=int(1600), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--T_max', type=int, default=int(1e5), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--replay_frequency', type=int, default=1, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--reward_clip', type=int, default=1, metavar='VALUE',
                        help='Reward clipping (0 to disable)')
    parser.add_argument('--target_update', type=int, default=int(2e3), metavar='τ',
                        help='Number of steps after which to update target network')

    # optimizer parameters
    parser.add_argument('--clip_value', type=float, default=10, metavar='NORM',
                        help='Max L2 norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε',
                        help='Adam epsilon')
    parser.add_argument('--optim_name', type=str, default='adam',
                        choices=['adam', 'lars'],
                        help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='LARS optimizer weight decay factor')

    # Evaluate parameter
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

    # cuda and seed
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Ables CUDA training (default: cuda:0)')
    parser.add_argument('--enable_cudnn', action='store_true',
                        help='Enable cuDNN (faster but nondeterministic)')
    seed = np.random.randint(12345)
    parser.add_argument('--seed', type=int, default=seed,
                        help='Random seed')

    return parser


def stdim_parser():
    parser = argparse.ArgumentParser(description='Stdim coupled with efficient Rainbow')

    # environment option
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

    # Rainbow parameter
    parser.add_argument('--architecture', type=str,
                        default='data_efficient', metavar='ARCH',
                        choices=['canonical', 'data_efficient'],
                        help='Network architecture')
    parser.add_argument('--V_min', type=float, default=-10, metavar='V',
                        help='Minimum of value distribution support')
    parser.add_argument('--V_max', type=float, default=10, metavar='V',
                        help='Maximum of value distribution support')

    # state option
    parser.add_argument('--resize', type=int, default=84,
                        help='Resize state information')

    # Noisy network parameter
    # FIXME 0.1 -> 0.5
    parser.add_argument('--noisy_std', type=float, default=0.5, metavar='σ',
                        help='Initial standard deviation of noisy linear layers')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='SIZE',
                        help='Network hidden size')

    # SimCLR & ST-DIM parameter
    parser.add_argument('--projection_size', type=int, default=128, metavar='SIZE',
                        help='Network hidden size')
    parser.add_argument('--local_depth', type=int, default=32, metavar='SIZE',
                        help='Feature map depth size')
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='Logit scaling factor')

    # Replay memory parameter
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

    # Training hyperparamters
    parser.add_argument('--multi_step', type=int, default=20, metavar='n',
                        help='Number of steps for multi-step return')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='γ',
                        help='Reward discount factor')
    parser.add_argument('--batch_size', type=int, default=32, metavar='SIZE',
                        help='Batch size')
    parser.add_argument('--model', type=str, metavar='PARAMS',
                        help='Pretrained model (state dict)')
    parser.add_argument('--learn_start', type=int, default=int(1600), metavar='STEPS',
                        help='Number of steps before starting training')
    parser.add_argument('--T_max', type=int, default=int(1e5), metavar='STEPS',
                        help='Number of training steps (4x number of frames)')
    parser.add_argument('--replay_frequency', type=int, default=1, metavar='k',
                        help='Frequency of sampling from memory')
    parser.add_argument('--reward_clip', type=int, default=1, metavar='VALUE',
                        help='Reward clipping (0 to disable)')
    parser.add_argument('--target_update', type=int, default=int(2e3), metavar='τ',
                        help='Number of steps after which to update target network')

    # optimizer parameters
    parser.add_argument('--clip_value', type=float, default=10, metavar='NORM',
                        help='Max L2 norm for gradient clipping')
    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='η',
                        help='Learning rate')
    parser.add_argument('--adam_eps', type=float, default=1.5e-4, metavar='ε',
                        help='Adam epsilon')
    parser.add_argument('--optim_name', type=str, default='adam',
                        choices=['adam', 'lars'],
                        help='Adam epsilon')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='LARS optimizer weight decay factor')

    # Evaluate parameter
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

    # Upper Confidence Bound Multi-Armed Bandit Problem parameter
    parser.add_argument('--ucb_exploration_coef', type=float, default=0.5,
                        help='UCB Exploration coefficient')
    parser.add_argument('--ucb_window_length', type=int, default=10,
                        help='Sliding Window Average of the Past K mean returns')
    parser.add_argument('--random_choice_step', type=int, default=10000,
                        help='N-step random choice for exploring UCB')

    # cuda and seed
    parser.add_argument('--cuda', type=str, default='cuda:0',
                        help='Ables CUDA training (default: cuda:0)')
    parser.add_argument('--enable_cudnn', action='store_true',
                        help='Enable cuDNN (faster but nondeterministic)')
    seed = np.random.randint(12345)
    parser.add_argument('--seed', type=int, default=seed,
                        help='Random seed')

    return parser
