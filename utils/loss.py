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


class SimCLRLoss(nn.Module):
    def __init__(self, args: argparse):
        super(SimCLRLoss, self).__init__()

        self.args = args

    def forward(self,
                features: torch.FloatTensor,
                n_views: int = 2):

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


class SupConLoss(nn.Module):
    """
    This code is modified Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR
    """
    def __init__(self, args: argparse):
        super(SupConLoss, self).__init__()

        self.args = args

    def forward(self,
                features: torch.FloatTensor,
                n_views: int = 2,
                labels: torch.LongTensor = None,
                mask: torch.Tensor = None):
        """
        Compute loss for model. If both 'labels' and 'mask' are None,
        it degenerates to SimCLR unsupervised loss: https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [batch_size * n_views, ...].
            n_views: Like SimCLR.
            labels: ground Truth of shape [batch_size].
            mask: contrastive mask of shape [batch_size, batch_size], mask_{i, j}=1 if sample j
                has the same class as sample i. Can be asymmetric.

        Returns:
            A loss scalar.
        """
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both labels and mask.')

        elif labels is None and mask is None:
            mask = torch.eye(self.args.batch_size, dtype=torch.float32).to(device=self.args.cuda)

        elif labels is not None:
            # Define appropriate labels for n_views
            labels = torch.cat([labels, labels], dim=0)

            labels = labels.contiguous().view(-1, 1)

            if labels.shape[0] != self.args.batch_size:
                raise ValueError('Number of labels does not match number of features.')

            mask = torch.eq(labels, labels.T).float().to(device=self.args.cuda)

        else:
            mask = mask.float().to(self.args.cuda)

        # Normalize feature vector
        features = F.normalize(features, dim=1)

        # Computes batched the p-norm distance between each pair of the two collections of row vectors
        distances = torch.cdist(features.detach(), features.detach(), p=2)

        # Define the top k items with the highest distance and make mask based on the distance.
        pos_idx = torch.topk(distances, self.args.pos_candidate).indices
        dist_mask = torch.scatter(
            torch.zeros_like(distances),
            dim=1,
            index=pos_idx,
            value=1)

        # Compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),  # Anchor dot Contrast
            self.args.scl_temperature)

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            dim=1,
            index=torch.arange(self.args.batch_size * n_views).view(-1, 1).to(device=self.args.cuda),
            value=0)

        # Define mask based on same action labels except self-contrast cases
        mask = mask * logits_mask

        # Compute log probability
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True))

        # Define our proposed method mask (same action labels and top k items with the highest distance)
        final_mask = mask * dist_mask

        # Compute mean of log-likelihood over positive
        mean_log_prob_pos = (final_mask * log_prob).sum(dim=1) / final_mask.sum(dim=1)

        # Supervised Contrastive Loss (SCL)
        loss = - (self.args.scl_temperature / self.args.base_scl_temperature) * mean_log_prob_pos
        loss = loss.view(n_views, self.args.batch_size).mean()

        return loss
