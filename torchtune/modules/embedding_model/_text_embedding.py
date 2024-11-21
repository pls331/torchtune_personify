# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import nn
from torchtune.modules import TransformerDecoder

class PoolingType:
    EOS = "eos"
    MEAN_POOLING = "mean"
    # 1.Muennighoff, N. SGPT: GPT Sentence Embeddings for Semantic Search. arXiv (2022) doi:10.48550/arxiv.2202.08904.
    POS_W_MEAN_POOLING = "pos_w_mean"

class TextEmbeddingTransformerDecoder(nn.Module):
    # TODO (pls331): add block comment for the class
    def __init__(
        self,
        decoder: TransformerDecoder,
        emb_pooling: str,
    ):
        super().__init__()
        self.decoder = decoder
        self.emb_pooling = emb_pooling
    
    def forward(
        self,
        tokens: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        #TODO(pls331): implement bi-directional mask
        #TODO(pls331): Masked Next Token Prediction - MNTP
        #TODO(pls331): SimCSE
        (
            h_last_layer, # [b, s, d]
            output, # [b, s, out_dim]
        ) = self.decoder(
            tokens=tokens,
            mask=mask,
            encoder_input=encoder_input,
            encoder_mask=encoder_mask,
            input_pos=input_pos,
        )

        if self.emb_pooling == PoolingType.EOS:
            pooled_emb = h_last_layer[:, -1, :] # [b, 1, d]
        elif self.emb_pooling == PoolingType.MEAN_POOLING:
            pooled_emb = torch.sum(h_last_layer, dim=1, keepdim=True)
        elif self.emb_pooling == PoolingType.POS_W_MEAN_POOLING:
            S = h_last_layer.shape(1)
            pos_w = torch.arange(S) / torch.sum(torch.arange(S))
            pooled_emb = torch.sum(
                h_last_layer * pos_w.reshape(1, -1, 1), # [b, s, d]
                dim=1,
                keepdim=True,
            )
        else:
            raise NotImplementedError(f"{self.emb_pooling=}")

        return pooled_emb.squeeze(1) # [b, d]

            

