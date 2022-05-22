import torch
import torch.nn as nn


def weights_init(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='linear')
        torch.nn.init.zeros_(m.bias)
