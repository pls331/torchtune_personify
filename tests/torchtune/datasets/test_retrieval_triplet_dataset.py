# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
from unittest.mock import patch

import pytest

from tests.test_utils import DummyTokenizer

from torchtune.datasets import msmarco_dataset


class TestRetrievalTripletDataset:
    @pytest.fixture
    def tokenizer(self):
        return DummyTokenizer()

    # @patch("torchtune.datasets._text_completion.load_dataset")
    # @pytest.mark.parametrize("max_seq_len", [128, 512, 1024, 4096])
    def test_dataset_get_item(self, load_dataset, tokenizer):
        # Sample data from wikitext dataset
        load_dataset.return_value = [
            {
                "query": "Who is the 1st US president?",
                "positive": "Gorge Washington",
                "negative": "Barack Obama",
            }
        ]
        ds = msmarco_dataset(
            tokenizer=tokenizer,
            max_seq_len=max_seq_len,
        )
        assert len(ds) > 0
        query, positive, negative = ds[0]["query"], ds[0]["positive"], ds[0]["negative"]
