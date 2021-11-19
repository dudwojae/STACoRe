from __future__ import division

import argparse
import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from utils.layers import weights_init


# Factorized NoisyLinear with bias
class NoisyLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 std_init: float = 0.5):
        super(NoisyLinear, self).__init__()

        self.module_name = 'noisy_linear'
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))

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

    def _scale_noise(self, size: int):
        x = torch.randn(size)

        return x.sign().mul_(x.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, input: torch.Tensor):
        if self.training:
            return F.linear(input, self.weight_mu + self.weight_sigma * self.weight_epsilon,
                            self.bias_mu + self.bias_sigma * self.bias_epsilon)

        else:
            return F.linear(input, self.weight_mu, self.bias_mu)


# Classifier for ST-DIM Calculation
class Classifier(nn.Module):
    def __init__(self,
                 feature1: int,
                 feature2: int):
        super().__init__()

        self.cls = nn.Bilinear(feature1, feature2, 1)

    def forward(self,
                x1: torch.Tensor,
                x2: torch.Tensor):

        return self.cls(x1, x2)


# MLP class for Self-Supervised Learning Projector & ST-DIM Global Head
class MLPHead(nn.Module):
    def __init__(self,
                 in_features: int,
                 projection_size: int):
        super(MLPHead, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_features, in_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features, projection_size, bias=True))

    def forward(self, x: torch.Tensor):
        return self.mlp(x)


# Q-value & Self-Supervised & SpatioTemporal
class ActSCoRe_DQN(nn.Module):
    def __init__(self,
                 args: argparse,
                 action_space: int):
        super(ActSCoRe_DQN, self).__init__()

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

        self.fc_h_v = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)
        self.fc_h_a = NoisyLinear(self.conv_output_size, args.hidden_size, std_init=args.noisy_std)

        self.fc_z_v = NoisyLinear(args.hidden_size, self.atoms, std_init=args.noisy_std)
        self.fc_z_a = NoisyLinear(args.hidden_size, action_space * self.atoms, std_init=args.noisy_std)

        self.W_h = nn.Parameter(torch.rand(self.conv_output_size, args.hidden_size))
        self.W_c = nn.Parameter(torch.rand(args.hidden_size, 128))

        self.b_h = nn.Parameter(torch.zeros(args.hidden_size))
        self.b_c = nn.Parameter(torch.zeros(128))

        self.W = nn.Parameter(torch.rand(128, 128))

        # First layer norm
        self.ln1 = nn.LayerNorm(args.hidden_size)  # 256

        # Second layer norm
        self.ln2 = nn.LayerNorm(args.hidden_size // 2)  # 128

        # MLP head (nn.Module)
        self.projector = MLPHead(self.conv_output_size, args.projection_size)

        # Network Initial weights
        self.apply(weights_init)

    def forward(self,
                x: torch.Tensor,
                log=False):

        x1 = self.convs[:2](x)
        x2 = self.convs[2:](x1)
        # x = self.convs(x)
        x3 = x2.view(-1, self.conv_output_size)  # Flatten

        # for RL
        v = self.fc_z_v(F.relu(self.fc_h_v(x3)))  # Value stream
        a = self.fc_z_a(F.relu(self.fc_h_a(x3)))  # Advantage stream

        v, a = v.view(-1, 1, self.atoms), a.view(-1, self.action_space, self.atoms)
        q = v + a - a.mean(1, keepdim=True)  # Combine streams

        if log:  # Use log softmax for numerical stability
            q = F.log_softmax(q, dim=2)  # Log probabilities with action over second dimension

        else:
            q = F.softmax(q, dim=2)  # Probabilities with action over second dimension

        # For Self-Supervised Learning Feature & ST-DIM Global feature
        if self.args.ssl_option == 'moco':
            proj_z = torch.matmul(x3, self.W_h) + self.b_h  # Contrastive head
            proj_z = F.relu(self.ln1(proj_z))
            proj_z = torch.matmul(proj_z, self.W_c) + self.b_c  # Contrastive head
            proj_z = self.ln2(proj_z)

        else:
            proj_z = self.projector(x3)

        return q, x2, proj_z  # Change x1 to x2

    def reset_noise(self):
        for name, module in self.named_children():

            if 'fc' in name:
                module.reset_noise()
