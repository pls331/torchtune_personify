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
    SUM = "sum"
    MEAN = "mean"
    # 1.Muennighoff, N. SGPT: GPT Sentence Embeddings for Semantic Search. arXiv (2022) doi:10.48550/arxiv.2202.08904.
    POS_W_MEAN = "pos_w_mean"

def get_pos_w(batch_seqlen: torch.Tensor, d: int, pooling_type:PoolingType):
    B = batch_seqlen.size(0)
    max_seqlen = torch.max(batch_seqlen).item()
    pos_w = torch.zeros((B, max_seqlen), dtype=torch.int, device=batch_seqlen.device)

    for i in range(B):
        S = batch_seqlen[i].item()
        if pooling_type == PoolingType.POS_W_MEAN: 
            pos_w[i, :S] = torch.arange(S) + 1
        else:
            pos_w[i, :S] = 1

    if pooling_type == PoolingType.SUM: 
        return pos_w
    else: # calculate ratio
        pos_w = pos_w / torch.sum(pos_w, dim=1, keepdim=True) # [B, S]
        return pos_w.unsqueeze(-1).expand(-1, -1, d) # [B, S, d]

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
        batch_seqlen: torch.Tensor,
        *,
        mask: Optional[torch.Tensor] = None,
        encoder_input: Optional[torch.Tensor] = None,
        encoder_mask: Optional[torch.Tensor] = None,
        input_pos: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        batch_seqlen: [B] seqlen len of each sample, to denote the range for pooling.
        """
        #TODO(pls331)-P0: implement bi-directional mask
        #TODO(pls331)-P0: Masked Next Token Prediction - MNTP
        #TODO(pls331)-P0: SimCSE
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

        B, S, d = h_last_layer.size()
        batch_seqlen = batch_seqlen.to(h_last_layer.device)

        # TODO(pls331): this only supports causal attention mask
        if self.emb_pooling == PoolingType.EOS:
            # TODO(pls331)-P0: implement EOS pooling properly
            # EOS is not always the last token because of padding
            pooled_emb = h_last_layer[:, -1, :]
        elif self.emb_pooling in {PoolingType.SUM, PoolingType.MEAN, PoolingType.POS_W_MEAN}:
            pos_w = get_pos_w(batch_seqlen, d, self.emb_pooling)
            h_last_layer = pos_w * h_last_layer
            pooled_emb = torch.sum(h_last_layer, dim=1, keepdim=True)
        else:
            raise NotImplementedError(f"{self.emb_pooling=}")

        return pooled_emb.squeeze(1) # [b, 1, d] -> [b, d]

            
class RandomEmbeddingModel(torch.nn.Module):
    def __init__(self, embed_dim=512) -> None:
        super().__init__()
        self.embd_dim: int = embed_dim

    def forward(
        self, tokens: torch.Tensor, batch_seqlen: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        return torch.randn(tokens.size(0), self.embd_dim)
