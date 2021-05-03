import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimCLRLoss(nn.Module):
    def __init__(self, args: argparse, temperature: float):
        super(SimCLRLoss, self).__init__()

        self.args = args
        self.temperature = temperature

    def forward(self, features: torch.FloatTensor):
        """For SimCLR"""

        _, num_views, _ = features.size()

        # Normalize features to lie on a unit hypershere
        features = F.normalize(features, dim=-1)
        features = torch.cat(torch.unbind(features, dim=1), dim=0)  # (B, N, F) -> (NB, F)
        contrasts = features

        # Compute logits (aka. similarity scores) & numerically stabilize them
        logits = features @ contrasts.T  # (NB, F) x (F, NB * world_size)
        logits = logits.div(self.temperature)

        # Compute masks
        _, pos_mask, neg_mask = self.create_masks(logits.size(), self.args, num_views)

        # Compute loss
        numerator = logits * pos_mask
        denominator = torch.exp(logits) * pos_mask.logical_or(neg_mask)
        denominator = denominator.sum(dim=1, keepdim=True)
        log_prob = numerator - torch.log(denominator)
        mean_log_prob = (log_prob * pos_mask) / pos_mask.sum(dim=1, keepdim=True)
        loss = torch.neg(mean_log_prob)
        loss = loss.sum(dim=1).mean()

        return loss, logits, pos_mask

    @staticmethod
    @torch.no_grad()
    def create_masks(shape, args: argparse, num_views: int = 2):

        device = args.cuda
        nL, nG = shape

        local_mask = torch.eye(nL // num_views, device=device).repeat(2, 2)  # self+positive indicator
        local_pos_mask = local_mask - torch.eye(nL, device=device)  # positive indicator
        local_neg_mask = torch.ones_like(local_mask) - local_mask  # negative indicator

        # Global mask of self+positive indicators
        global_mask = torch.zeros(nL, nG, device=device)
        global_mask[:, nL * 0 * (0 + 1)] = local_mask

        # Global mask of positive indicators
        global_pos_mask = torch.zeros_like(global_mask)
        global_pos_mask[:, nL * 0 * (0 + 1)] = local_pos_mask

        # Global mask of negative indicators
        global_neg_mask = torch.ones_like(global_mask)
        global_neg_mask[:, nL * 0 * (0 + 1)] = local_neg_mask

        return global_mask, global_pos_mask, global_neg_mask
