# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from typing import List
from functools import partial

from torchtune.models.llama3_2_embedding._component_builders import llama3_2_text_encoder

from torchtune.modules import TransformerDecoder

def llama3_2_1b_text_encoder() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 1b parameter values.
    
    Returns:
        TransformerDecoder: Instantiation of Llama3.2 1B model
    """
    return llama3_2_text_encoder(
        vocab_size=128_256,
        num_layers=16,
        num_heads=32,
        num_kv_heads=8,
        embed_dim=2048,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )

def llama3_2_3b_text_encoder() -> TransformerDecoder:
    """
    Builder for creating a Llama3.2 model initialized w/ the default 3b parameter values.

    Returns:
        TransformerDecoder: Instantiation of Llama3.2 3B model
    """
    return llama3_2_text_encoder(
        vocab_size=128_256,
        num_layers=28,
        num_heads=24,
        num_kv_heads=8,
        embed_dim=3072,
        max_seq_len=131072,
        intermediate_dim=8192,
        attn_dropout=0.0,
        norm_eps=1e-5,
        rope_base=500_000,
        scale_factor=32,
    )