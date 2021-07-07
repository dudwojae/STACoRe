import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F


class STDIMLoss(nn.Module):
    def __init__(self, args: argparse):
        super(STDIMLoss, self).__init__()

        self.args = args

    def forward(self,
                global_t: torch.Tensor,
                local_t_map: torch.Tensor,
                local_t_prev_map: torch.Tensor,
                local_t_n_map: torch.Tensor):

        # Permute (B, C, H, W) -> (B, W, H, C)
        local_t_map = local_t_map.permute(0, 3, 2, 1)
        local_t_prev_map = local_t_prev_map.permute(0, 3, 2, 1)
        local_t_n_map = local_t_n_map.permute(0, 3, 2, 1)

        # Loss 1: Global-Local
        global_t = global_t.unsqueeze(1).unsqueeze(1). \
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


class MoCoLoss(nn.Module):
    def __init__(self, args: argparse):
        super(MoCoLoss, self).__init__()

        self.args = args

    def forward(self, network_weights,
                online_features: torch.FloatTensor,
                target_features: torch.FloatTensor):

        online_features = F.normalize(online_features, dim=1)
        target_features = F.normalize(target_features, dim=1)

        z_proj = torch.matmul(network_weights, target_features.T)
        logits = torch.matmul(online_features, z_proj)
        logits = (logits - torch.max(logits, 1)[0][:, None])
        logits = logits * 0.1
        labels = torch.arange(logits.shape[0]).long().to(device=self.args.cuda)

        return logits, labels


class BYOLLoss(nn.Module):
    def __init__(self, args: argparse):
        super(BYOLLoss, self).__init__()

        self.args = args

    def forward(self, augmentation1: nn.Module,
                augmentation2: nn.Module,
                states: torch.FloatTensor,
                next_states: torch.FloatTensor):

        # Apply augmentation & Switch
        states_aug1 = augmentation1.mixing(states).to(device=self.args.cuda)
        next_states_aug2 = augmentation2.mixing(next_states).to(device=self.args.cuda)

        # For Switch
        states_aug2 = next_states_aug2
        next_states_aug1 = states_aug1

        return states_aug1, states_aug2, next_states_aug1, next_states_aug2

    @staticmethod
    def calculate_loss(pred, true):
        pred = F.normalize(pred, dim=1)
        true = F.normalize(true, dim=1)

        return 2 - 2 * (pred * true).sum(dim=-1)


class SimCLRLoss(nn.Module):
    def __init__(self, args: argparse):
        super(SimCLRLoss, self).__init__()

        self.args = args

    def forward(self, features: torch.FloatTensor, n_views: int = 2):

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
