from __future__ import division

import os
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn

import kornia.augmentation as aug

from networks.actscore_network import ActSCoRe_DQN, Classifier
from utils.automatic import UCBAugmentation
from utils.loss import STDIMLoss, ActSCoReLoss


class ActSCoReAgent:
    def __init__(self,
                 args: argparse,
                 env,
                 result_path: str):

        self.args = args
        self.result_path = result_path
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max

        # Support (range) of z
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.cuda)
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.coeff = args.lambda_coef if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.

        # Define Model (Default: Off-Policy Reinforcement Learning)
        self.online_net = ActSCoRe_DQN(args, self.action_space).to(device=args.cuda)
        self.target_net = ActSCoRe_DQN(args, self.action_space).to(device=args.cuda)

        # Load Pre-trained Model If Provided
        if args.model:

            if os.path.isfile(args.model):
                # Always load tensors onto CPU by default, will shift to GPU if necessary
                state_dict = torch.load(args.model, map_location='cpu')

                if 'conv1.weight' in state_dict.keys():
                    for old_key, new_key in (('conv1.weight', 'convs.0.weight'),
                                             ('conv1.bias', 'convs.0.bias'),
                                             ('conv2.weight', 'convs.2.weight'),
                                             ('conv2.bias', 'convs.2.bias'),
                                             ('conv3.weight', 'convs.4.weight'),
                                             ('conv3.bias', 'convs.4.bias')):
                        state_dict[new_key] = state_dict[old_key]  # Re-map state dict for old pretrained models
                        del state_dict[old_key]  # Delete old keys for strict load_state_dict

                self.online_net.load_state_dict(state_dict)
                print(f"Loading pretrained model: {args.model}")

            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        # Initial Data Augmentation (Random Crop)
        self.init_aug = nn.Sequential(aug.RandomCrop((80, 80)),
                                      nn.ReplicationPad2d(4),
                                      aug.RandomCrop((args.resize, args.resize)))

        # Define UCB Multi-Armed Bandit Problem (Automatic Augmentation)
        if self.args.ucb_option:
            self.ucb = UCBAugmentation(args=args)
            self.aug_func_list = self.ucb.aug_func_list

        # Define ST-DIM & Classifier (classifier1 = Global, Local / classifier2 = Local, Local)
        if self.args.stcl_option == 'stdim':
            self.stdim_loss = STDIMLoss(args=args)

            self.classifier1 = Classifier(args.projection_size,
                                          args.local_depth).to(device=args.cuda)  # input size (128, 64)
            self.classifier2 = Classifier(args.local_depth,
                                          args.local_depth).to(device=args.cuda)  # input size (64, 64)

            self.spatiotemporal_on = True

        else:
            self.spatiotemporal_on = False

        # Define Supervised Contrastive Learning Methods
        if self.args.ssl_option == 'actscore':
            self.actscore_loss = ActSCoReLoss(args=args)

            self.ssl_on = True

        else:
            self.ssl_on = False

        # Define Optimizer
        if self.spatiotemporal_on:  # ST-DIM
            self.optim_params = list(self.online_net.parameters()) + \
                                list(self.classifier1.parameters()) + \
                                list(self.classifier2.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate, eps=args.adam_eps)

            # Train mode
            self.classifier1.train()
            self.classifier2.train()

        else:  # Not ST-DIM
            self.optim_params = list(self.online_net.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate, eps=args.adam_eps)

        # RL Online & Target Network (Default: Off-Policy Reinforcement Learning)
        self.online_net.train()
        self.update_target_net()
        self.target_net.train()
        for param_t in self.target_net.parameters():
            param_t.requires_grad = False

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state: torch.Tensor):
        with torch.no_grad():
            self.online_net.eval()
            a, _, _ = self.online_net(state.unsqueeze(0))

        return (a * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_epsilon_greedy(self,
                           state: torch.Tensor,
                           epsilon: float = 0.001):

        # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) \
            if np.random.random() < epsilon else self.act(state)

    # For RL target network
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def optimize(self,
                 memory,
                 timesteps: int):

        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = \
            memory.sample(self.args.batch_size)

        # Apply initial Data Augmentation to States (Random Crop) for SSL
        init_states1 = self.init_aug(states).to(device=self.args.cuda)
        init_states2 = self.init_aug(states).to(device=self.args.cuda)

        # Apply initial Data Augmentation to Next States (Random Crop) for ST-DIM
        init_states = self.init_aug(states).to(device=self.args.cuda)
        init_next_states = self.init_aug(next_states).to(device=self.args.cuda)

        # Multi-Armed Bandit
        if self.spatiotemporal_on:

            if not self.args.ucb_option and not self.ssl_on:  # Baseline
                print('We are experimenting with the baseline model.')

            elif self.args.ucb_option and self.ssl_on:  # Proposed Method
                self.current_aug_id, aug_list = self.ucb.select_ucb_aug(timestep=timesteps)
                aug_sequential = nn.Sequential(*aug_list)

                aug_states1 = aug_sequential[0](init_states1)
                aug_states2 = aug_sequential[1](init_states2)

            elif not self.args.ucb_option and self.ssl_on:  # Ablation Studies
                aug_states1 = init_states1
                aug_states2 = init_states2

            else:
                raise NotImplementedError('We need to make the UCB and SSL switch mode the same.')

        else:
            raise NotImplementedError('ST-DIM switch is off...')

        if self.args.ssl_option == 'actscore':
            _, _, supcon_z = self.online_net(torch.cat([aug_states1, aug_states2], dim=0), log=True)

            ssl_loss, num_positives = self.actscore_loss(features=supcon_z,
                                                         labels=actions.detach())

        if self.args.stcl_option == 'stdim':
            # ST-DIM Update
            _, feature_map_t, stdim_z = self.online_net(init_states, log=True)  # anchor
            _, feature_map_t1, _ = self.online_net(init_next_states, log=True)  # positives

            # negative examples -> shuffle states
            shuffle_states = torch.stack(
                random.sample(list(init_next_states), len(init_next_states))
            )
            _, feature_map_tn, _ = self.online_net(shuffle_states, log=True)

            x1, x2, x1_l, x2_l, target = self.stdim_loss(global_t=stdim_z,
                                                         local_t_map=feature_map_t,
                                                         local_t_prev_map=feature_map_t1,
                                                         local_t_n_map=feature_map_tn)
            stdim_gl_loss = (nn.BCEWithLogitsLoss()
                             (self.classifier1(x1, x2).squeeze(), target)).to(device=self.args.cuda)
            stdim_ll_loss = (nn.BCEWithLogitsLoss()
                             (self.classifier2(x1_l, x2_l).squeeze(), target)).to(device=self.args.cuda)
            stcl_loss = stdim_gl_loss + stdim_ll_loss

        # RL update
        log_ps, _, _ = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.args.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate n-th next state probabilities
            pns, _, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))

            # Perform argmax action selection using online network argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)

            # Sample new target net noise
            self.target_net.reset_noise()

            pns, _, _ = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)

            # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)
            pns_a = pns[range(self.args.batch_size), argmax_indices_ns]

            # Compute Tz (Bellman operator T applied to z)
            # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = returns.unsqueeze(1) + nonterminals * \
                 (self.args.gamma ** self.args.multi_step) * self.support.unsqueeze(0)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values

            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)

            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.args.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.args.batch_size - 1) * self.atoms),
                                    self.args.batch_size).unsqueeze(1).expand(
                self.args.batch_size, self.atoms).to(actions)

            # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (l + offset).view(-1), (pns_a * (u.float() - b)).view(-1))
            # m_u = m_u + p(s_t+n, a*)(b - l)
            m.view(-1).index_add_(0, (u + offset).view(-1), (pns_a * (b - l.float())).view(-1))

        rl_loss = -torch.sum(m * log_ps_a, 1)

        if self.spatiotemporal_on and self.ssl_on:  # Proposed Method
            loss = rl_loss + (self.coeff * (ssl_loss + stcl_loss))

        elif self.spatiotemporal_on and not self.ssl_on:  # Baseline
            loss = rl_loss + (self.coeff * stcl_loss)

        self.optimizer.zero_grad()
        total_loss = (weights * loss).mean()
        total_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        # Clip gradients by L2 norm
        torch.nn.utils.clip_grad_norm_(self.optim_params, self.args.clip_value)
        self.optimizer.step()

        # Save SSL Loss with RL Loss & Save number of action labels
        if timesteps % 100 == 0:
            if self.spatiotemporal_on and self.ssl_on:  # Proposed Method
                self.log_loss_action(f'| Reinforcement Learning = {rl_loss.mean().item()}'
                                     f'| Self-Supervised Learning = {ssl_loss.item()}'
                                     f'| SpatioTemporal Contrastive Learning = {stcl_loss.item()}'
                                     f'| Batch of Action Labels = {actions}'
                                     f'| Number of Positives = {sum(num_positives) / 4096}')

            elif self.spatiotemporal_on and not self.ssl_on:  # Baseline
                self.log_loss_action(f'| Reinforcement Learning = {rl_loss.mean().item()}'
                                     f'| SpatioTemporal Contrastive Learning = {stcl_loss.item()}'
                                     f'| Batch of Action Labels = {actions}')

        # Update priorities of sampled transitions
        memory.update_priorities(idxs, loss.detach().cpu().numpy())

        # Update Upper Confidence Bound with SSL Loss
        if self.args.ucb_option:
            self.ucb.update_ucb_values(augmentation_id=self.current_aug_id,
                                       batch_returns=ssl_loss)

    # Save model parameters on current device (don't move model between devices)
    def save(self,
             path: str,
             name: str = 'rainbow.pt'):

        torch.save(self.online_net.state_dict(), os.path.join(path, name))

        if self.args.ucb_option:
            # Save Augmentation function list as CSV
            auglist = pd.DataFrame(self.aug_func_list, columns=['aug1', 'aug2'])
            auglist.to_csv(os.path.join(path, 'AugCombination.csv'), index=True)

    def log_loss_action(self,
                        s: str,
                        name='loss_and_action.txt'):

        filename = os.path.join(self.result_path, name)

        if not os.path.exists(filename) or s is None:
            f = open(filename, 'w')

        else:
            f = open(filename, 'a')

        f.write(str(s) + '\n')
        f.close()

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        a, _, _ = self.online_net(state.unsqueeze(0))

        return (a * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
