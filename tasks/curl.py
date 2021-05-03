from __future__ import division

import os
import bz2
import pickle
import numpy as np

from tqdm import trange
from datetime import datetime

from utils.summary import TensorboardSummary
from utils.memory import ReplayMemory
from environment.env import Atari_Env

from singleagents.curl_agent import RainbowAgent
from tasks.curl_test import test


class Curl_Rainbow:
    def __init__(self, args, result_path):
        self.args = args
        self.result_path = result_path

        # Define Tensorboard summary
        self.summary = TensorboardSummary(result_path)
        self.writer = self.summary.create_summary()

        # Define Atari environment
        self.env = Atari_Env(args)
        self.env.train()
        self.action_space = self.env.action_space()

        # Define Curl Rainbow Agent
        self.learner = RainbowAgent(args, self.env)

        # Define metrics
        self.metrics = {'steps': [], 'rewards': [], 'Qs': [], 'best_avg_reward': -float('inf')}

    def run_curl_rainbow(self):
        # If a model is provided, and evaluate is fale, presumably we want to resume, so try to load memory
        if self.args.model is not None and not self.args.evaluate:

            if not self.args.memory:
                raise ValueError('Cannot resume training without memory save path. Aborting...')

            elif not os.path.exists(self.args.memory):
                raise ValueError(f'Could not find memory file at {self.args.memory}. Aborting...')

            memory = self.load_memory(self.args.memory, self.args.disable_bzip_memory)

        else:
            memory = ReplayMemory(self.args, self.args.memory_capacity)

        priority_weight_increase = (1 - self.args.priority_weight) / \
                                   (self.args.T_max - self.args.learn_start)

        # Construct validation memory
        val_memory = ReplayMemory(self.args, self.args.evaluation_size)
        T, done = 0, True
        while T < self.args.evaluation_size:

            if done:
                state, done = self.env.reset(), False

            next_state, _, done = self.env.step(np.random.randint(0, self.action_space))
            val_memory.append(state, None, None, done)
            state = next_state
            T += 1

        if self.args.evaluate:
            # Set DQN (online network) to evaluation mode
            self.learner.eval()
            avg_reward, avg_Q = test(self.args,
                                     0,
                                     self.learner,
                                     val_memory,
                                     self.metrics,
                                     self.result_path,
                                     evaluate=True)
            print(f'Avg. reward: {str(avg_reward)} | Avg. Q: {str(avg_Q)}')

        else:
            # Training loop
            self.learner.train()
            T, done = 0, True
            for T in trange(1, self.args.T_max + 1):

                if done:
                    state, done = self.env.reset(), False

                if T % self.args.replay_frequency == 0:
                    self.learner.reset_noise()  # Draw a new set of noisy weights

                action = self.learner.act(state)
                next_state, reward, done = self.env.step(action)  # Step

                if self.args.reward_clip > 0:
                    # Clip rewards
                    reward = max(min(reward, self.args.reward_clip), -self.args.reward_clip)

                memory.append(state, action, reward, done)  # Append transition to memory

                # Train and test
                if T >= self.args.learn_start:
                    # Anneal importance sampling weight β to 1
                    memory.priority_weight = min(memory.priority_weight + priority_weight_increase, 1)

                    if T % self.args.replay_frequency == 0:
                        # Train with n-step distributional double Q-learning
                        self.learner.optimize(memory)
                        # MoCo momentum update
                        self.learner.update_momentum_net()

                    if T % self.args.evaluation_interval == 0:
                        # Set DQN (online network) to evaluation mode
                        self.learner.eval()
                        # Test
                        avg_reward, avg_Q = test(self.args,
                                                 T,
                                                 self.learner,
                                                 val_memory,
                                                 self.metrics,
                                                 self.result_path)
                        self.log(f'T = {str(T)} / {str(self.args.T_max)} '
                                 f'| Avg.reward: {str(avg_reward)} | Avg. Q: {str(avg_Q)}')
                        # Set DQN (online network) back to training mode
                        self.learner.train()

                        # If memory path provided, save it
                        if self.args.memory is not None:
                            self.save_memory(memory, self.args.memory, self.args.disable_bzip_memory)

                    # Update target network
                    if T % self.args.target_update == 0:
                        self.learner.update_target_net()

                    # Checkpoint the network
                    if (self.args.checkpoint_interval != 0) and (T % self.args.checkpoint_interval == 0):
                        self.learner.save(self.result_path)

                state = next_state

        self.env.close()

    @staticmethod
    def log(s):
        print(f"[{str(datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))}] {s}")

    @staticmethod
    def load_memory(memory_path, disable_bzip):
        if disable_bzip:
            with open(memory_path, 'rb') as pickle_file:
                return pickle.load(pickle_file)

        else:
            with open(memory_path, 'rb') as zipped_pickle_file:
                return pickle.load(zipped_pickle_file)

    @staticmethod
    def save_memory(memory, memory_path, disable_bzip):
        if disable_bzip:
            with open(memory_path, 'wb') as pickle_file:
                pickle.dump(memory, pickle_file)

        else:
            with bz2.open(memory_path, 'wb') as zipped_pickle_file:
                pickle.dump(memory, zipped_pickle_file)



