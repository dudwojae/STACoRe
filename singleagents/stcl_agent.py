from __future__ import division
import time
import os
import random
import argparse
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F

from networks.stcl_network import STCL_DQN, Classifier, MLPHead
from utils.augmentations import StateAugMix
from utils.automatic import UCBAugmentation
from utils.loss import STDIMLoss, MoCoLoss, BYOLLoss, SimCLRLoss


class STCLAgent:
    def __init__(self, args: argparse, env, result_path: str):
        self.args = args
        self.result_path = result_path
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max

        # Support (range) of z
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.cuda)
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.coeff = 0.01 if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.

        # Define Model (Default: Off-Policy Reinforcement Learning)
        self.online_net = STCL_DQN(args, self.action_space).to(device=args.cuda)
        self.target_net = STCL_DQN(args, self.action_space).to(device=args.cuda)

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

        # (Optional) Define UCB Multi-Armed Bandit Problem (Automatic Augmentation)
        if self.args.ucb_option:
            self.ucb = UCBAugmentation(args=args)
            self.aug_func_list = self.ucb.aug_func_list

        # Define StateAugMix Module
        self.stateaugmix = StateAugMix(args=args)

        # Define ST-DIM & Classifier (classifier1 = Global, Local / classifier2 = Local, Local)
        if self.args.stcl_option == 'stdim':
            self.stdim_loss = STDIMLoss(args=args)

            self.classifier1 = Classifier(args.projection_size,
                                          args.local_depth).to(device=args.cuda)  # input size (128, 64)
            self.classifier2 = Classifier(args.local_depth,
                                          args.local_depth).to(device=args.cuda)  # input size (64, 64)

            self.spatiotemporal_on = True

        elif self.args.stcl_option == 'stcl':  # FIXME: Our New Algorithm
            self.spatiotemporal_on = True
            pass

        else:
            raise NotImplementedError

        # Define Self-Supervised Contrastive Learning Methods
        if self.args.ssl_option == 'moco':
            self.moco_loss = MoCoLoss(args=args)

            self.momentum_net = STCL_DQN(args, self.action_space).to(device=args.cuda)

            # Momentum Network Train Mode
            self.initialize_momentum_net()
            self.momentum_net.train()

            for m_param in self.momentum_net.parameters():
                m_param.requires_grad = False

            self.ssl_on = True
            self.byol_on = False

        elif self.args.ssl_option == 'byol':
            self.byol_loss = BYOLLoss(args=args)

            self.momentum_net = STCL_DQN(args, self.action_space).to(device=args.cuda)
            # Define BYOL Predictor Model
            self.byol_predictor = MLPHead(in_features=args.projection_size,
                                          projection_size=args.projection_size).to(device=args.cuda)

            # Momentum Network Train Mode
            self.initialize_momentum_net()
            self.momentum_net.train()

            for m_param in self.momentum_net.parameters():
                m_param.requires_grad = False

            self.ssl_on = True
            self.byol_on = True  # For Predictor MLPHead Gradient Descent

        elif self.args.ssl_option == 'simclr':
            self.simclr_loss = SimCLRLoss(args=args)

            self.ssl_on = True
            self.byol_on = False

        else:
            self.ssl_on = False
            self.byol_on = False

        # Define Optimizer
        if self.spatiotemporal_on and self.byol_on:  # ST-DIM & BYOL Experiment
            self.optim_params = list(self.online_net.parameters()) + \
                                list(self.classifier1.parameters()) + \
                                list(self.classifier2.parameters()) + \
                                list(self.byol_predictor.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate, eps=args.adam_eps)

            # Train mode
            self.classifier1.train()
            self.classifier2.train()
            self.byol_predictor.train()

        elif self.spatiotemporal_on and not self.byol_on:  # ST-DIM & MoCo or SimCLR or None
            self.optim_params = list(self.online_net.parameters()) + \
                                list(self.classifier1.parameters()) + \
                                list(self.classifier2.parameters())
            self.optimizer = torch.optim.Adam(self.optim_params,
                                              lr=args.learning_rate, eps=args.adam_eps)

            # Train mode
            self.classifier1.train()
            self.classifier2.train()

        else:
            raise NotImplementedError

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
    def act_epsilon_greedy(self, state: torch.Tensor, epsilon: float = 0.001):
        # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) \
            if np.random.random() < epsilon else self.act(state)

    # For MoCo & BYOL
    def initialize_momentum_net(self):
        for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(param_q.data)  # update
            param_k.requires_grad = False  # not update by gradient

    # For RL target network
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.no_grad()
    def update_momentum_net(self):
        """
        Exponential Moving Average Update (Same as MoCo Momentum Update)
        """
        momentum = self.args.momentum
        for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

    def optimize(self, memory, timesteps: int):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = \
            memory.sample(self.args.batch_size)

        # (Optional) Multi-Armed Bandit
        if self.args.ucb_option and self.ssl_on:
            self.current_aug_id, aug_list = self.ucb.select_ucb_aug(timestep=timesteps)

            aug_states = aug_list[0](states).to(device=self.args.cuda)
            aug_next_states = aug_list[1](next_states).to(device=self.args.cuda)

            mixed_states = self.stateaugmix.mixing(aug_states)
            mixed_next_states = self.stateaugmix.mixing(aug_next_states)

        elif not self.args.ucb_option and self.ssl_on:
            mixed_states = self.stateaugmix.mixing(states)
            mixed_next_states = self.stateaugmix.mixing(next_states)

        else:
            # Choose Implementation Data Augmentations for ST-DIM (Without SSL)
            if self.args.stcl_aug_on:
                mixed_states = self.stateaugmix.mixing(states)
                mixed_next_states = self.stateaugmix.mixing(next_states)

            else:
                mixed_states = states
                mixed_next_states = next_states

        if self.args.ssl_option == 'moco':
            _, _, z_anchor = self.online_net(mixed_states, log=True)
            _, _, z_target = self.momentum_net(mixed_next_states, log=True)
            logits, labels = self.moco_loss(network_weights=self.online_net.W,
                                            online_features=z_anchor,
                                            target_features=z_target)

            ssl_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.cuda)

        elif self.args.ssl_option == 'byol':
            states_aug1, states_aug2, \
            next_states_aug1, next_states_aug2 = self.byol_loss(augmentation1=self.stateaugmix,
                                                                augmentation2=self.stateaugmix,
                                                                states=states,
                                                                next_states=next_states)
            # Compute online features
            _, _, projections_states_aug1 = self.online_net(states_aug1, log=True)
            _, _, projections_states_aug2 = self.online_net(states_aug2, log=True)
            # State t
            predictions_states_aug1 = self.byol_predictor(projections_states_aug1)
            # State t with different augmentation
            predictions_states_aug2 = self.byol_predictor(projections_states_aug2)

            # Compute byol target features
            with torch.no_grad():
                # State t + 1
                _, _, target_next_states_aug2 = self.momentum_net(next_states_aug2, log=True)
                # State t + 1 with different augmentation
                _, _, target_next_states_aug1 = self.momentum_net(next_states_aug1, log=True)

            # Calculate byol loss
            loss = self.byol_loss.calculate_loss(predictions_states_aug1, target_next_states_aug2)
            loss += self.byol_loss.calculate_loss(predictions_states_aug2, target_next_states_aug1)

            ssl_loss = loss.mean()

        elif self.args.ssl_option == 'simclr':
            _, _, simclr_z = self.online_net(torch.cat([mixed_states, mixed_next_states], dim=0), log=True)
            logits, labels = self.simclr_loss(features=simclr_z)

            ssl_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.cuda)

        if self.args.stcl_option == 'stdim':
            # ST-DIM Update
            _, feature_map_t, stdim_z = self.online_net(mixed_states, log=True)  # anchor
            _, feature_map_t1, _ = self.online_net(mixed_next_states, log=True)  # positives

            # negative examples -> shuffle states
            shuffle_states = torch.stack(
                random.sample(list(mixed_next_states), len(mixed_next_states))
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

        elif self.args.stcl_option == 'stcl':  # FIXME: Our New Algorithm
            pass

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

        if self.spatiotemporal_on and self.ssl_on:
            loss = rl_loss + (self.coeff * (ssl_loss + stcl_loss))

        elif self.spatiotemporal_on and not self.ssl_on:
            loss = rl_loss + (self.coeff * stcl_loss)

        self.optimizer.zero_grad()
        total_loss = (weights * loss).mean()
        total_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        # Clip gradients by L2 norm  # FIXME
        torch.nn.utils.clip_grad_norm_(self.optim_params, self.args.clip_value)
        self.optimizer.step()

        # Save SSL Loss with RL Loss
        if timesteps % 1000 == 0:
            if self.spatiotemporal_on and self.ssl_on:
                self.log_loss(f'Reinforcement Learning = {rl_loss.mean().item()}'
                              f'| Self-Supervised Learning = {ssl_loss.item()}'
                              f'| SpatioTemporal Contrastive Learning = {stcl_loss.item()}')

            elif self.spatiotemporal_on and not self.ssl_on:
                self.log_loss(f'Reinforcement Learning = {rl_loss.mean().item()}'
                              f'| SpatioTemporal Contrastive Learning = {stcl_loss.item()}')

        # Update priorities of sampled transitions
        memory.update_priorities(idxs, loss.detach().cpu().numpy())

        # (Optional) Update Upper Confidence Bound with SimCLR Loss, RL Loss
        if self.args.ucb_option:
            self.ucb.update_ucb_values(augmentation_id=self.current_aug_id,
                                       batch_returns=rl_loss.mean() + ssl_loss)

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='stdim_rainbow.pt'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

        if self.args.ucb_option:
            # (Optional) Save Augmentation function list as CSV
            auglist = pd.DataFrame(self.aug_func_list, columns=['aug1', 'aug2'])
            auglist.to_csv(os.path.join(path, 'AugCombination.csv'), index=True)

    def log_loss(self, s, name='loss.txt'):
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
