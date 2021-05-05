from __future__ import division

import os
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque

import kornia.augmentation as aug

from networks.stdim_network import STDIM_DQN, Classifier


class STDIMAgent:
    def __init__(self, args: argparse, env):
        self.args = args
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max

        # Support (range) of z
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.cuda)
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.coeff = 0.01 if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.

        # Define model
        self.online_net = STDIM_DQN(args, self.action_space).to(device=args.cuda)
        self.target_net = STDIM_DQN(args, self.action_space).to(device=args.cuda)

        # Define Classifier for ST-DIM (classifier1 = Global, Local / classifier2 = Local, Local)
        self.classifier1 = Classifier(args.projection_size,
                                      args.local_depth).to(device=args.cuda)  # input size (128, 64)
        self.classifier2 = Classifier(args.local_depth,
                                      args.local_depth).to(device=args.cuda)  # input size (64, 64)

        # Image-based augmentation option (Upper Confidence Bound)
        self.augmentations = {
            'RandomCrop': nn.Sequential(aug.RandomCrop((80, 80)),
                                        nn.ReplicationPad2d(4),
                                        aug.RandomCrop((84, 84))),
            'CenterCrop': nn.Sequential(aug.CenterCrop((84, 84))),
            'CutOut': nn.Sequential(aug.RandomErasing(p=0.5)),
            'Rotation': nn.Sequential(aug.RandomRotation(degrees=5.0)),
            'HorizontalFlip': nn.Sequential(aug.RandomHorizontalFlip(p=0.1)),
            'VerticalFlip': nn.Sequential(aug.RandomVerticalFlip(p=0.1)),
        }
        self.aug_func_list = [v for k, v in self.augmentations.items()]

        self.num_augmentations = len(self.augmentations)  # 6
        self.total_num = 1
        self.num_action = [1.] * self.num_augmentations
        self.qval_action = [0.] * self.num_augmentations

        self.expl_action = [0.] * self.num_augmentations
        self.ucb_action = [0.] * self.num_augmentations

        self.return_action = []
        for i in range(self.num_augmentations):
            self.return_action.append(deque(maxlen=args.ucb_window_length))

        # Load pretrained model if provided
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

        # Define optimizer
        optim_params = list(self.online_net.parameters()) + \
                       list(self.classifier1.parameters()) + \
                       list(self.classifier2.parameters())
        self.optimizer = torch.optim.Adam(optim_params,
                                          lr=args.learning_rate, eps=args.adam_eps)

        # Train mode
        self.online_net.train()
        self.classifier1.train()
        self.classifier2.train()

        # RL target network
        self.update_target_net()
        self.target_net.train()
        for param_t in self.target_net.parameters():
            param_t.requires_grad = False

    # Automatic data augmentation (Upper Confidence Bound)
    def select_ucb_aug(self, timestep: int):

        for i in range(self.num_augmentations):
            self.expl_action[i] = self.args.ucb_exploration_coef * \
                                  np.sqrt(np.log(self.total_num) / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]

        if timestep < self.args.random_choice_step:  # Random choice
            ucb_aug_ids = np.random.choice(np.argsort(self.ucb_action),
                                           size=2,
                                           replace=False)  # Two augmentations, No Duplication
            ucb_aug_id1, ucb_aug_id2 = ucb_aug_ids[1], ucb_aug_ids[0]

        else:
            ucb_aug_ids = np.argsort(self.ucb_action)[-2:]  # Two augmentations
            ucb_aug_id1, ucb_aug_id2 = ucb_aug_ids[1], ucb_aug_ids[0]

        return ucb_aug_id1, ucb_aug_id2, \
            self.aug_func_list[ucb_aug_id1], self.aug_func_list[ucb_aug_id2]

    def update_ucb_values(self, batch_returns: torch.FloatTensor):
        # Twice Calculation
        self.total_num += 1
        self.num_action[self.current_aug_id1] += 1
        self.num_action[self.current_aug_id2] += 1

        self.return_action[self.current_aug_id1].append(batch_returns.mean().item())
        self.return_action[self.current_aug_id2].append(batch_returns.mean().item())

        self.qval_action[self.current_aug_id1] = np.mean(self.return_action[self.current_aug_id1])
        self.qval_action[self.current_aug_id2] = np.mean(self.return_action[self.current_aug_id2])

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

    # For RL target network
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # SimCLR loss & update
    def simclr_loss(self, features: torch.FloatTensor, n_views: int = 2):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(device=self.args.cuda)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)

        # Discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(device=self.args.cuda)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)

        # Select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # Select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(device=self.args.cuda)

        logits = logits / self.args.temperature

        return logits, labels

    # ST-DIM loss & update
    def stdim_loss(self,
                   global_t: torch.Tensor,
                   local_t_map: torch.Tensor,
                   local_t_prev_map: torch.Tensor,
                   local_t_n_map: torch.Tensor):

        # Permute (B, C, H, W) -> (B, W, H, C)
        local_t_map = local_t_map.permute(0, 3, 2, 1)
        local_t_prev_map = local_t_prev_map.permute(0, 3, 2, 1)
        local_t_n_map = local_t_n_map.permute(0, 3, 2, 1)

        # Loss 1: Global-Local
        global_t = global_t.unsqueeze(1).unsqueeze(1).\
            expand(-1, local_t_prev_map.size(1), local_t_prev_map.size(2), self.args.projection_size)

        target = torch.cat((torch.ones_like(global_t[:, :, :, 0]),
                            torch.zeros_like(global_t[:, :, :, 0])), dim=0).to(device=self.args.cuda)

        x1 = torch.cat([global_t, global_t], dim=0)
        x2 = torch.cat([local_t_prev_map, local_t_n_map], dim=0)

        shuffled_idxs = torch.randperm(len(target))
        x1, x2, target = x1[shuffled_idxs], x2[shuffled_idxs], target[shuffled_idxs]

        # Loss 2: Local-Local
        x1_l = torch.cat([local_t_map, local_t_map], dim=0)
        x2_l = torch.cat([local_t_prev_map, local_t_n_map], dim=0)

        x1_l, x2_l = x1_l[shuffled_idxs], x2_l[shuffled_idxs]

        return x1, x2, x1_l, x2_l, target

    def optimize(self, memory, timesteps: int):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = \
            memory.sample(self.args.batch_size)

        self.current_aug_id1, self.current_aug_id2, aug1, aug2 = self.select_ucb_aug(timestep=timesteps)

        # SimCLR update
        aug_states_1 = aug1(states).to(device=self.args.cuda)
        aug_states_2 = aug2(states).to(device=self.args.cuda)

        _, _, simclr_z = self.online_net(torch.cat([aug_states_1, aug_states_2], dim=0))

        logits, labels = self.simclr_loss(simclr_z)
        simclr_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.cuda)

        # ST-DIM update
        _, feature_map_t, stdim_z = self.online_net(states)  # anchor
        _, feature_map_t1, _ = self.online_net(next_states)  # positives

        # negative examples -> shuffle states
        shuffle_states = torch.stack(
            random.sample(list(next_states), len(next_states))
        )
        _, feature_map_tn, _ = self.online_net(shuffle_states)

        x1, x2, x1_l, x2_l, target = self.stdim_loss(global_t=stdim_z,
                                                     local_t_map=feature_map_t,
                                                     local_t_prev_map=feature_map_t1,
                                                     local_t_n_map=feature_map_tn)
        stdim_gl_loss = (nn.BCEWithLogitsLoss()
                         (self.classifier1(x1, x2).squeeze(), target)).to(device=self.args.cuda)
        stdim_ll_loss = (nn.BCEWithLogitsLoss()
                         (self.classifier2(x1_l, x2_l).squeeze(), target)).to(device=self.args.cuda)
        stdim_loss = stdim_gl_loss + stdim_ll_loss

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

        loss = -torch.sum(m * log_ps_a, 1)
        loss = loss + (self.coeff * (simclr_loss + stdim_loss))

        self.optimizer.zero_grad()
        total_loss = (weights * loss).mean()
        total_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        # Clip gradients by L2 norm
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.args.clip_value)
        self.optimizer.step()

        # Update Upper Confidence Bound
        self.update_ucb_values(returns)

        # Update priorities of sampled transitions
        memory.update_priorities(idxs, loss.detach().cpu().numpy())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='stdim_rainbow.pt'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

        # Evaluates Q-value based on single state (no batch)

    def evaluate_q(self, state):
        a, _, _ = self.online_net(state.unsqueeze(0))

        return (a * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
