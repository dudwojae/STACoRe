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

        self.aug_func_list = []
        for v1 in self.augmentations.values():
            for v2 in self.augmentations.values():
                self.aug_func_list.append([v1, v2])  # All Combination

        self.num_augmentations = len(self.aug_func_list)  # 36

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

    def update_ucb_values(self, augmentation_id: int,
                          batch_returns: torch.FloatTensor):

        batch_returns = batch_returns.detach().cpu().numpy()

        # Twice Calculation
        self.total_num += 1

        self.num_action[augmentation_id] += 1
        self.return_action[augmentation_id].append(-batch_returns.item())
        self.qval_action[augmentation_id] = np.mean(self.return_action[augmentation_id])
