import argparse
import random
import numpy as np

import torch
import torch.nn as nn
from torch.distributions.dirichlet import Dirichlet
from torch.distributions.beta import Beta

import kornia.augmentation as aug


class StateMix(nn.Module):
    def __init__(self, args: argparse):
        super(StateMix, self).__init__()
        self.args = args

    def forward(self,
                prev_state: torch.FloatTensor,
                next_state: torch.FloatTensor):

        # Bounding Box coordinate
        bbx_11, bbx_21, bby_11, bby_21, \
        bbx_12, bbx_22, bby_12, bby_22 = self.random_bbox(prev_state.size())

        prev_state[:, :, bbx_11:bbx_21, bby_11:bby_21], \
        next_state[:, :, bbx_12:bbx_22, bby_12:bby_22] \
            = next_state[:, :, bbx_11:bbx_21, bby_11:bby_21], \
              prev_state[:, :, bbx_12:bbx_22, bby_12:bby_22]

        return prev_state, next_state

    def random_bbox(self, size: tuple):
        lambda_1 = np.random.beta(self.args.alpha, self.args.alpha)
        lambda_2 = np.random.beta(self.args.alpha, self.args.alpha)

        width = size[2]
        height = size[3]

        cut_ratio_1 = np.sqrt(1. - lambda_1)

        cut_width_1 = np.int(width * cut_ratio_1)
        cut_height_1 = np.int(height * cut_ratio_1)

        # uniform (location)
        cut_x_1 = np.random.randint(width)
        cut_y_1 = np.random.randint(height)

        # Bounding Box coordinate
        bbx_11 = np.clip(cut_x_1 - cut_width_1 // 2, 0, width)
        bbx_21 = np.clip(cut_x_1 + cut_width_1 // 2, 0, width)

        bby_11 = np.clip(cut_y_1 - cut_height_1 // 2, 0, height)
        bby_21 = np.clip(cut_y_1 + cut_height_1 // 2, 0, height)

        cut_ratio_2 = np.sqrt(1. - lambda_2)

        cut_width_2 = np.int(width * cut_ratio_2)
        cut_height_2 = np.int(height * cut_ratio_2)

        # uniform (location)
        cut_x_2 = np.random.randint(width)
        cut_y_2 = np.random.randint(height)

        # Bounding Box coordinate
        bbx_12 = np.clip(cut_x_2 - cut_width_2 // 2, 0, width)
        bbx_22 = np.clip(cut_x_2 + cut_width_2 // 2, 0, width)

        bby_12 = np.clip(cut_y_2 - cut_height_2 // 2, 0, height)
        bby_22 = np.clip(cut_y_2 + cut_height_2 // 2, 0, height)

        return bbx_11, bbx_21, bby_11, bby_21, \
               bbx_12, bbx_22, bby_12, bby_22


class StateAugMix:
    def __init__(self, args: argparse):
        super(StateAugMix, self).__init__()

        self.args = args
        # Kornia Augmentation Version
        self.augmentations = {
            'Equalize': aug.RandomEqualize(p=1.0),
            'Posterize': aug.RandomPosterize(bits=(0, 8), p=1.0),
            'Solarize': aug.RandomSolarize(thresholds=(0., 1.0), p=1.0),
            'Rotation': aug.RandomRotation(degrees=15., p=1.0),
            'Translate_X': aug.RandomAffine(translate=(0.1, 0.), shear=None,
                                            scale=None, p=1.0, degrees=0.),
            'Translate_Y': aug.RandomAffine(translate=(0., 0.1), shear=None,
                                            scale=None, p=1.0, degrees=0.),
            'Shear_X': aug.RandomAffine(shear=(0., 0.3, 0., 0.), translate=None,
                                        scale=None, p=1.0, degrees=0.),
            'Shear_Y': aug.RandomAffine(shear=(0., 0., 0., 0.3), translate=None,
                                        scale=None, p=1.0, degrees=0.)
        }
        self.augmentations_list = [v for v in self.augmentations.values()]

    def mixing(self, original: torch.FloatTensor):
        x_temp = original  # Back Up for Skip Connection

        x_aug = torch.zeros_like(original)
        mixing_weight_dist = Dirichlet(torch.empty(self.args.k).
                                       fill_(self.args.alpha).to(device=self.args.cuda))
        mixing_weights = mixing_weight_dist.sample()

        for i in range(self.args.k):
            sampled_augs = random.sample(self.augmentations_list, self.args.k)
            aug_chain_length = random.choice(range(1, self.args.k + 1))
            aug_chain = sampled_augs[:aug_chain_length]

            aug_sequential = nn.Sequential(*aug_chain)

            x_temp = aug_sequential(x_temp)

            x_aug += mixing_weights[i] * x_temp

            # Reset
            x_temp = original

        skip_conn_weight_dist = Beta(torch.tensor([self.args.alpha]).to(device=self.args.cuda),
                                     torch.tensor([self.args.alpha]).to(device=self.args.cuda))
        skip_conn_weight = skip_conn_weight_dist.sample()

        x_augmix = skip_conn_weight * x_aug + (1 - skip_conn_weight) * original

        return x_augmix


'''
StateAugMix Old Version
class StateAugMix:
    def __init__(self, args: argparse):
        super(StateAugMix, self).__init__()

        self.args = args
        self.augmentations = [self.equalize, self.posterize, self.rotate,
                              self.solarize, self.shear_x, self.shear_y,
                              self.translate_x, self.translate_y]

    def mixing(self, original: torch.FloatTensor):
        original = original.detach().cpu()
        x_temp = original  # Back Up for Skip Connection

        x_aug = torch.zeros_like(original)
        mixing_weight_dist = Dirichlet(torch.empty(self.args.k).fill_(self.args.alpha))
        mixing_weights = mixing_weight_dist.sample()

        for i in range(self.args.k):
            sampled_augs = random.sample(self.augmentations, self.args.k)
            aug_chain_length = random.choice(range(1, self.args.k + 1))
            aug_chain = sampled_augs[:aug_chain_length]

            for aug in aug_chain:
                severity = random.choice(range(1, 6))  # Default Number
                x_temp = aug(x_temp, severity)

            x_aug += mixing_weights[i] * x_temp

        skip_conn_weight_dist = Beta(torch.tensor([self.args.alpha]),
                                     torch.tensor([self.args.alpha]))
        skip_conn_weight = skip_conn_weight_dist.sample()

        x_augmix = skip_conn_weight * x_aug + (1 - skip_conn_weight) * original

        return x_augmix

    # Kornia Version
    @staticmethod
    def int_parameter(level: int, maxval: int):

        """Helper function to scale `val` between 0 and maxval .
        Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
        level/PARAMETER_MAX.
        Returns:
        An int that results from scaling `maxval` according to `level`.
        """

        return int(level * maxval / 10)

    @staticmethod
    def float_parameter(level: int, maxval: int):

        """Helper function to scale `val` between 0 and maxval.
        Args:
        level: Level of the operation that will be between [0, `PARAMETER_MAX`].
        maxval: Maximum value that the operation can have. This will be scaled to
          level/PARAMETER_MAX.
        Returns:
        A float that results from scaling `maxval` according to `level`.
        """

        return float(level) * maxval / 10.

    @staticmethod
    def sample_level(n: float):

        return np.random.uniform(low=0.1, high=n)

    def equalize(self, state: torch.FloatTensor, _):

        return kornia.enhance.equalize(state)

    def posterize(self, state: torch.FloatTensor, level: int):
        level = self.int_parameter(self.sample_level(level), 8)

        return kornia.enhance.posterize(state, bits=(8 - level))

    def solarize(self, state: torch.FloatTensor, level: int):
        level = self.float_parameter(self.sample_level(level), 1)

        return kornia.enhance.solarize(state, thresholds=(1.0 - level))

    def rotate(self, state: torch.FloatTensor, level: int):
        degrees = self.int_parameter(self.sample_level(level), 30)

        if np.random.uniform() > 0.5:
            degrees = -degrees

        return kornia.geometry.rotate(state, angle=torch.FloatTensor([degrees]), mode='bilinear')

    def shear_x(self, state: torch.FloatTensor, level: int):
        level = self.float_parameter(self.sample_level(level), 0.3)

        if np.random.uniform() > 0.5:
            level = -level

        sheared_x = torch.tensor([[level, 0.0]])
        return kornia.geometry.shear(state, shear=sheared_x)

    def shear_y(self, state: torch.FloatTensor, level: int):
        level = self.float_parameter(self.sample_level(level), 0.3)

        if np.random.uniform() > 0.5:
            level = -level

        sheared_y = torch.tensor([[0.0, level]])
        return kornia.geometry.shear(state, shear=sheared_y)

    def translate_x(self, state: torch.FloatTensor, level: int):
        level = self.int_parameter(self.sample_level(level), self.args.resize / 6)  # 84 means state size

        if np.random.random() > 0.5:
            level = -level

        translation_x = torch.tensor([[level, 0.0]])
        return kornia.geometry.translate(state, translation=translation_x)

    def translate_y(self, state: torch.FloatTensor, level: int):
        level = self.int_parameter(self.sample_level(level), self.args.resize / 6)  # 84 means state size

        if np.random.random() > 0.5:
            level = -level

        translation_y = torch.tensor([[0.0, level]])
        return kornia.geometry.translate(state, translation=translation_y)
'''