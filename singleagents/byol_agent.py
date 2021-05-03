from __future__ import division

import os
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

import kornia.augmentation as aug

from utils.augmentations import Intensity
from networks.byol_network import BYOL_DQN, MLPHead


class BYOLAgent:
    def __init__(self, args, env):
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
        self.online_net = BYOL_DQN(args, self.action_space).to(device=args.cuda)
        self.byol_target_net = BYOL_DQN(args, self.action_space).to(device=args.cuda)
        self.target_net = BYOL_DQN(args, self.action_space).to(device=args.cuda)

        # Define BYOL predictor model
        self.predictor = MLPHead(in_features=args.projection_size,
                                 mlp_hidden_size=args.byol_hidden_size,
                                 projection_size=args.projection_size).to(device=args.cuda)

        # Image-based augmentation option (random crop & center crop)
        self.aug1 = nn.Sequential(aug.RandomCrop((80, 80)),
                                  nn.ReplicationPad2d(4),
                                  aug.RandomCrop((84, 84)))

        # FIXME: Other data augmentation
        # self.aug2 = nn.Sequential(aug.CenterCrop((84, 84)))  # Center crop
        # self.aug2 = nn.Sequential(aug.RandomCrop((80, 80)),
        #                           nn.ReplicationPad2d(4),
        #                           aug.RandomCrop((84, 84)))  # Random crop
        # self.aug2 = nn.Sequential(aug.RandomErasing(p=0.5))  # Cutout
        self.aug2 = nn.Sequential(aug.RandomHorizontalFlip(p=0.1))  # Horizontal Flip
        # self.aug2 = nn.Sequential(aug.RandomVerticalFlip(p=0.1))  # Vertical Flip
        # self.aug2 = nn.Sequential(aug.RandomRotation(degrees=5.0))  # Rotation
        # self.aug2 = nn.Sequential(nn.ReplicationPad2D(4),
        #                           aug.RandomCrop((84, 84)))  # Random Shift
        # self.aug2 = nn.Sequential(aug.ColorJitter(0.2, 0.3, 0.2, 0.3, same_on_batch=True))  # Color Jitter
        # self.aug2 = Intensity(scale=0.05)  # Intensity

        if args.model:  # Load pretrained model if provided

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

        self.online_net.train()

        # BYOL target network
        self.initialize_byol_target_net()
        self.byol_target_net.train()

        # RL target network
        self.update_target_net()
        self.target_net.train()
        for param_b, param_t in zip(self.byol_target_net.parameters(), self.target_net.parameters()):
            param_b.requires_grad, param_t.requires_grad = False, False

        # Define optimizer
        self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                          lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            self.online_net.eval()  # FIXME
            a, _ = self.online_net(state.unsqueeze(0))

            return (a * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_epsilon_greedy(self, state, epsilon=0.001):
        # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) \
            if np.random.random() < epsilon else self.act(state)

    # For byol target network
    def initialize_byol_target_net(self):
        for param_q, param_k in zip(self.online_net.parameters(), self.byol_target_net.parameters()):
            param_k.data.copy_(param_q.data)  # Initialize
            param_k.requires_grad = False  # Not update by gradient

    # For RL target network
    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    @torch.no_grad()
    def update_byol_target_net(self):
        """
        Exponential moving average update (same as MOCO momentum update)
        """
        momentum = self.args.momentum
        for param_q, param_k in zip(self.online_net.parameters(), self.byol_target_net.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

    @staticmethod
    def byol_loss(pred, true):
        pred = F.normalize(pred, dim=1)
        true = F.normalize(true, dim=1)

        return 2 - 2 * (pred * true).sum(dim=-1)

    def byol_update(self, states, next_states):
        # Apply augmentation & switch
        states_aug1 = self.aug1(states).to(device=self.args.cuda)
        next_states_aug2 = self.aug2(next_states).to(device=self.args.cuda)

        # For switch
        states_aug2 = self.aug2(states).to(device=self.args.cuda)
        next_states_aug1 = self.aug1(next_states).to(device=self.args.cuda)

        # Compute online features
        _, projections_states_aug1 = self.online_net(states_aug1)
        _, projections_states_aug2 = self.online_net(states_aug2)
        predictions_states_aug1 = self.predictor(projections_states_aug1)  # State t
        predictions_states_aug2 = self.predictor(projections_states_aug2)  # State t with different augmentation

        # Compute byol target features
        with torch.no_grad():
            _, target_next_states_aug2 = self.byol_target_net(next_states_aug2)  # State t + 1
            _, target_next_states_aug1 = self.byol_target_net(next_states_aug1)  # State t + 1 with different augmentation

        # Calculate byol loss
        loss = self.byol_loss(predictions_states_aug1, target_next_states_aug2)
        loss += self.byol_loss(predictions_states_aug2, target_next_states_aug1)

        return loss.mean()

    def optimize(self, memory):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = \
            memory.sample(self.args.batch_size)

        # BYOL update
        byol_loss = self.byol_update(states, next_states)

        # For RL update
        log_ps, _ = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.args.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate n-th next state probabilities
            pns, _ = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))

            # Perform argmax action selection using online network argmax_a[(z, p(s_t+n, a; θonline))]
            argmax_indices_ns = dns.sum(2).argmax(1)

            # Sample new target net noise
            self.target_net.reset_noise()

            pns, _ = self.target_net(next_states)  # Probabilities p(s_t+n, ·; θtarget)

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

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimizes DKL(m||p(s_t, a_t)))
        loss = loss + (self.coeff * byol_loss)

        self.optimizer.zero_grad()
        total_loss = (weights * loss).mean()
        total_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        # Clip gradients by L2 norm
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.args.clip_value)
        self.optimizer.step()

        # Update priorities of sampled transitions
        memory.update_priorities(idxs, loss.detach().cpu().numpy())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='byol_rainbow.pt'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        a, _ = self.online_net(state.unsqueeze(0))

        return (a * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
