import argparse
import numpy as np

import torch
import torch.nn as nn

from collections import deque

import kornia.augmentation as aug


class UCBAugmentation(object):
    def __init__(self, args: argparse):
        super(UCBAugmentation, self).__init__()

        self.args = args

        # Image-based augmentation option (Upper Confidence Bound)
        # Delete CutOut Operation
        self.augmentations = {
            'Affine': aug.RandomAffine(degrees=5.,
                                       translate=(0.14, 0.14),
                                       scale=(0.9, 1.1),
                                       shear=(-5, 5),
                                       p=0.5),
            'Translate': aug.RandomAffine(translate=(0.14, 0.14),
                                          shear=None,
                                          scale=None,
                                          p=0.5,
                                          degrees=0.),
            'Shear': aug.RandomAffine(shear=(-5, 5),
                                      translate=None,
                                      scale=None, p=0.5,
                                      degrees=0.),
            'Shift': nn.Sequential(nn.ReplicationPad2d(4),
                                   aug.RandomCrop((args.resize, args.resize))),
            'Intensity': Intensity(scale=0.05)
        }

        # All Data Augmentation Pair Combination
        self.aug_func_list = []
        for v1 in self.augmentations.values():
            for v2 in self.augmentations.values():
                self.aug_func_list.append([v1, v2])

        # Total Number is 25
        self.num_augmentations = len(self.aug_func_list)

        # UCB Update Parameters
        self.total_num = 1
        self.num_action = [1.] * self.num_augmentations
        self.qval_action = [0.] * self.num_augmentations

        self.expl_action = [0.] * self.num_augmentations
        self.ucb_action = [0.] * self.num_augmentations

        self.return_action = []
        for i in range(self.num_augmentations):
            self.return_action.append(deque(maxlen=args.ucb_window_length))

    # Automatic data augmentation (Upper Confidence Bound)
    def select_ucb_aug(self, timestep: int):

        for i in range(self.num_augmentations):
            self.expl_action[i] = self.args.ucb_exploration_coef * \
                                  np.sqrt(np.log(self.total_num) / self.num_action[i])
            self.ucb_action[i] = self.qval_action[i] + self.expl_action[i]

        if timestep < self.args.random_choice_step:  # Random choice
            ucb_aug_id = np.random.choice(np.arange(0, self.num_augmentations))

        else:
            ucb_aug_id = np.argmax(self.ucb_action)  # Two augmentations list

        return ucb_aug_id, self.aug_func_list[ucb_aug_id]

    def update_ucb_values(self,
                          augmentation_id: int,
                          batch_returns: torch.FloatTensor):

        batch_returns = batch_returns.detach().cpu().numpy()

        # Twice Calculation
        self.total_num += 1

        self.num_action[augmentation_id] += 1
        self.return_action[augmentation_id].append(-batch_returns.item())
        self.qval_action[augmentation_id] = np.mean(self.return_action[augmentation_id])


class Intensity(nn.Module):
    def __init__(self, scale: float):
        super().__init__()

        self.scale = scale

    def forward(self, x: torch.Tensor):
        r = torch.randn((x.size(0), 1, 1, 1), device=x.device)
        noise = 1.0 + (self.scale * r.clamp(-2.0, 2.0))

        return x * noise
