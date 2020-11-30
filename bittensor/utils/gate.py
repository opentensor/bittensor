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
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.projection = nn.Linear(x_dim, key_dim, bias=True).to(self.device)

    def forward(self, x: torch.Tensor, keys: torch.Tensor, topk: int):
        if x.dim() != 2:
            raise ValueError(
                'Ensure the context used for key routing is rank 2: x.dim() == {} != 2'
                .format(x.dim()))
        if x.size(1) != self.x_dim:
            raise ValueError(
                'Ensure that x.size(1) == x_dim:  x.size(1) === {} != {}'.
                format(x.size(1), self.x_dim))

        bs = x.shape[0]  # batch_size

        # Make input projection to k_dim dimension.
        query = self.projection(x)  # [bs, key_dim]
        # Check the query shape.
        if query.shape != (bs, self.key_dim):
            error_msg = 'The router projection returned an incorrect shape: {} != {}. Ensure that your projection is onto the keydim.'.format(
                query.shape, [(bs, self.key_dim)])
            raise ValueError(error_msg)

        query = query.view(-1, self.key_dim).to(self.device)  # (bs, key_dim)
        bs = query.shape[0]
        real_topk = min(keys.shape[0], topk)

        # Compute scores over keys.
        scores = F.linear(query, keys.to(self.device),
                          bias=None).to(self.device)  # (bs, n_keys)
        topk_scores, topk_indices = scores.topk(real_topk, dim=1)  # (bs, knn)

        zeros = torch.zeros(bs, keys.shape[0]).to(self.device)
        gates = zeros.scatter(1, topk_indices.to(self.device),
                              topk_scores.to(self.device))

        softmax = nn.Softmax(dim=1).to(self.device)
        gates = softmax(gates)

        return gates, scores
