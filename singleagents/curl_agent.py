from __future__ import division

import os
import numpy as np

import torch
import torch.nn as nn

import kornia.augmentation as aug

from networks.curl_network import DQN


class RainbowAgent:
    def __init__(self, args, env):
        self.args = args
        self.action_space = env.action_space()
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.cuda)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.coeff = 0.01 if args.game in ['pong', 'boxing', 'private_eye', 'freeway'] else 1.

        # Image-based augmentation option
        self.aug = nn.Sequential(aug.RandomCrop((80, 80)),
                                 nn.ReplicationPad2d(4),
                                 aug.RandomCrop((84, 84)))

        self.online_net = DQN(args, self.action_space).to(device=args.cuda)
        self.momentum_net = DQN(args, self.action_space).to(device=args.cuda)
        self.target_net = DQN(args, self.action_space).to(device=args.cuda)

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

        self.initialize_momentum_net()
        self.momentum_net.train()

        self.update_target_net()
        self.target_net.train()
        for m_param, t_param in zip(self.momentum_net.parameters(), self.target_net.parameters()):
            m_param.requires_grad, t_param.requires_grad = False, False

        self.optimizer = torch.optim.Adam(self.online_net.parameters(),
                                          lr=args.learning_rate, eps=args.adam_eps)

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state):
        with torch.no_grad():
            a, _ = self.online_net(state.unsqueeze(0))

            return (a * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_epsilon_greedy(self, state, epsilon=0.001):
        # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) \
            if np.random.random() < epsilon else self.act(state)

    def initialize_momentum_net(self):
        for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(param_q.data)  # update
            param_k.requires_grad = False  # not update by gradient

    def update_target_net(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Code for this function from https://github.com/facebookresearch/moco
    @torch.no_grad()
    def update_momentum_net(self, momentum=0.001):
        for param_q, param_k in zip(self.online_net.parameters(), self.momentum_net.parameters()):
            param_k.data.copy_(momentum * param_k.data + (1. - momentum) * param_q.data)

    def optimize(self, memory):
        # Sample transitions
        idxs, states, actions, returns, next_states, nonterminals, weights = \
            memory.sample(self.args.batch_size)
        aug_states_1 = self.aug(states).to(device=self.args.cuda)
        aug_states_2 = self.aug(states).to(device=self.args.cuda)

        # Calculate current state probabilities (online network noise already sampled)
        log_ps, _ = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        _, z_anchor = self.online_net(aug_states_1, log=True)
        _, z_target = self.momentum_net(aug_states_2, log=True)

        z_proj = torch.matmul(self.online_net.W, z_target.T)
        logits = torch.matmul(z_anchor, z_proj)
        logits = (logits - torch.max(logits, 1)[0][:, None])
        logits = logits * 0.1
        labels = torch.arange(logits.shape[0]).long().to(device=self.args.cuda)
        moco_loss = (nn.CrossEntropyLoss()(logits, labels)).to(device=self.args.cuda)

        log_ps_a = log_ps[range(self.args.batch_size), actions]  # log p(s_t, a_t; θonline)

        with torch.no_grad():
            # Calculate nth next state probabilities
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
        loss = loss + (moco_loss * self.coeff)

        self.optimizer.zero_grad()  # FIXME: online_net --> optimizer
        curl_loss = (weights * loss).mean()
        curl_loss.mean().backward()  # Backpropagate importance-weighted minibatch loss
        # Clip gradients by L2 norm
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), self.args.clip_value)
        self.optimizer.step()

        # Update priorities of sampled transitions
        memory.update_priorities(idxs, loss.detach().cpu().numpy())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path, name='curl_rainbow.pt'):
        torch.save(self.online_net.state_dict(), os.path.join(path, name))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        a, _ = self.online_net(state.unsqueeze(0))

        return (a * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
