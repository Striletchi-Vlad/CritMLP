import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from crit_functions import optimal_aspect_ratio, gamma_from_ratio,\
    distribution_hyperparameters


class CritMLP(nn.Module):
    def activation(self, x, af, neg_slope=None, pos_slope=None):
        if af == 'relu':
            return F.relu(x)
        if af == 'tanh':
            return F.tanh(x)
        if af == 'gelu':
            return F.GELU(x)
        if af == 'swish':
            return F.Hardswish(x)
        if af == 'linear':
            return x
        if af == 'relu-like':
            if neg_slope is None:
                neg_slope = 0
            if pos_slope is None:
                pos_slope = 1
            return neg_slope * x * (x < 0) + pos_slope * x * (x >= 0)

    def __init__(self, in_dim=None, out_dim=None, depth=None, width=None,
                 af='relu', neg_slope=None, pos_slope=None):
        super(CritMLP, self).__init__()

        if depth is not None and width is not None:
            raise ValueError('Either depth or width must be None, CritMLP \
                will infer the other to ensure criticality')

        if out_dim is None:
            raise ValueError('out_dim must be specified')

        if in_dim is None:
            raise ValueError('in_dim must be specified')

        if af != 'relu-like' and (neg_slope is not None or
                                  pos_slope is not None):
            raise ValueError('neg_slope and pos_slope are only used \
                for relu-like activation functions')

        self.af = af
        self.neg_slope = neg_slope
        self.pos_slope = pos_slope

        ratio = optimal_aspect_ratio(0, out_dim, af)
        self.gamma = gamma_from_ratio(ratio, out_dim, af)
        if depth is None:
            depth = int(width * ratio)
        if width is None:
            width = int(depth / ratio)

        layers = []
        layers.append(nn.Linear(in_dim, width))
        for i in range(depth-2):
            layers.append(nn.Linear(width, width))
        layers.append(nn.Linear(width, out_dim))

        def init_weights(m):
            Cb, CW = distribution_hyperparameters(
                self.gamma, af, neg_slope, pos_slope)
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=np.sqrt(CW/width))
                nn.init.normal_(m.bias, mean=0, std=np.sqrt(Cb))

        self.seq = nn.Sequential(*layers)
        # self.seq.apply(init_weights)

    def forward(self, x):
        for i in range(len(self.seq)):
            res = x
            x = self.seq[i](x)
            if i != len(self.seq)-1:
                x = self.activation(x, self.af, self.neg_slope, self.pos_slope)
                if i != 0:
                    x = x + res * self.gamma
                    pass
        return x
