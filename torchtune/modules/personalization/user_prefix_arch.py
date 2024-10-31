# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

class UserPrefixArch(nn.Module):
    def __init__(
        self, 
        num_user: int,
        n_user_token: int,
        emb_dim: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        self.n_user_token = n_user_token
        self.ffn_dim = ffn_dim
        self.emb_dim = emb_dim
        # TODO: implement parallelism based on fairscale:
        # 1) VocabParallelEmbedding (mp over vocab dim)
        # 2) ParallelEmbedding (mp over emb dim)
        self.total_emb_dim: int = self.n_user_token * emb_dim 
        self.user_prefix_emb = nn.Embedding(num_user, self.total_emb_dim)
        # TODO: MLP initialization?
        self.mlp: nn.Sequential = nn.Sequential(
            nn.Linear(self.total_emb_dim, ffn_dim),
            nn.Tanh(),
            nn.Linear(ffn_dim, self.total_emb_dim),
        )

    def forward(self, user_idxs: torch.Tensor) -> torch.Tensor:
        user_prefix_emb = self.user_prefix_emb(user_idxs)  # [B, n_user_token * emb_dim]
        B = user_idxs.shape[0]
        assert user_prefix_emb.shape == (
            B,
            1,
            self.total_emb_dim,
        ), (
            f"{user_prefix_emb.shape=}, {user_idxs.shape=}, {self.n_user_token=}," 
            f"{self.total_emb_dim=}, {self.ffn_dim=}, {self.emb_dim=}"
        )
        out = self.mlp(user_prefix_emb).view(B, self.n_user_token, -1)
        assert out.shape == (B, self.n_user_token, self.emb_dim)
        return out