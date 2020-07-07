import math
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class Gate(nn.Module):
    def __init__(self, x_dim, topk, key_dim):
        super().__init__()
        self.x_dim = x_dim
        self.key_dim = key_dim
        self.projection = nn.Linear(x_dim, key_dim, bias=True)

    def forward(self, x: torch.Tensor, keys: torch.Tensor, topk: int):
        assert topk >= len(keys)
        assert x.dim() == 2 and x.size(1) == self.x_dim
        bs = x.shape[0]  # batch_size

        # Make input projection to k_dim dimension.
        query = self.projection(x)  # [bs, key_dim]
        assert query.shape == (bs, self.key_dim)

        query = query.view(-1, self.key_dim)  # (bs, key_dim)
        bs = query.shape[0]
        assert query.dim() == 2 and query.size(1) == self.key_dim
        real_topk = min(keys.shape[0], topk)
        # Compute scores over keys.
        scores = F.linear(query, keys, bias=None)  # (bs, n_keys)
        topk_scores, topk_indices = scores.topk(real_topk, dim=1)  # (bs, knn)

        zeros = torch.zeros(bs, keys.shape[0])
        gates = zeros.scatter(1, topk_indices, topk_scores)

        softmax = nn.Softmax(dim=1)
        gates = softmax(gates)

        return gates
