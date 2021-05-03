from __future__ import division

import math

import torch
import torch.nn as nn
from torch.nn import functional as F


# BYOL + RAINBOW
# Factorized NoisyLinear with bias
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, std_init=0.5):
        super(NoisyLinear, self).__init__()

        self.module_name = 'noisy_linear'
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        # weights
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

        # biases
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)

        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)

        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


# MLP class for BYOL projector and predictor
class MLPHead(nn.Module):
    def __init__(self, in_features, mlp_hidden_size, projection_size):
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, mlp_hidden_size),
            nn.BatchNorm1d(mlp_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_size, projection_size)
        )

    def forward(self, x):
        return self.mlp(x)


# Q-value & BYOL projector
class BYOL_DQN(nn.Module):
    def __init__(self, args, action_space):
        super(BYOL_DQN, self).__init__()
        self.args = args
        self.atoms = args.atoms
        self.action_space = action_space

        if args.architecture == 'canonical':
            self.convs = nn.Sequential(
                nn.Conv2d(args.history_length, 32, kernel_size=8, stride=4, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
                nn.ReLU()
            )

            self.conv_output_size = 3136  # 64 * 7 * 7

        elif args.architecture == 'data_efficient':
            self.convs = nn.Sequential(
                nn.Conv2d(args.history_length, 32, kernel_size=5, stride=5, padding=0),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=5, stride=5, padding=0),
                nn.ReLU()
            )

            self.conv_output_size = 576  # 64 * 3 * 3

        # For Q-value (value, advantage)
        self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)

        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

        self.projector = MLPHead(self.conv_output_size, args.byol_hidden_size, args.projection_size)

    def forward(self, x, log=False):
        x = self.convs(x)
        x = x.view(-1, self.conv_output_size)  # Flatten

        # For RL
        v = self.fc_z_v(F.relu(self.fc_h_v(x)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x)))  # Advantage stream

        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension

        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension

        # For BYOL
        proj_h = self.projector(x)

        return q, proj_h

    def reset_noise(self):
        for name, module in self.named_children():

            if 'fc' in name:
                module.reset_noise()
