# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
import itertools
import random
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import torch
import mteb
import numpy as np
import tqdm

from mteb.encoder_interface import PromptType
from mteb.model_meta import ModelMeta
from mteb.encoder_interface import Encoder
from mteb.models.wrapper import Wrapper
from omegaconf import DictConfig, OmegaConf
from torchtune import config, training, utils
from torchtune.data import load_image, Message, padded_collate_tiled_images_and_mask
from torchtune.generation import sample
from torchtune.modules.tokenizers._utils import ModelTokenizer
from torchtune.modules.transforms import Transform


class MTEBModelWrapper(Encoder, Wrapper):
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: Transform,
        use_query_inst_format: bool,
        device: torch.device,
    ) -> None:
        self.model = model
        # model.meta = ModelMeta(
        #     name=None,
        #     revision=None,
        #     release_date=None,
        #     languages=None,
        # )
        self.tokenizer = tokenizer
        self._device = device
        self.use_query_inst_format = use_query_inst_format
        self._task_name_2_inst_task_defs = {
            "msmarco": [
                "Given a web search query, retrieve relevant passages that answer the query",
                "Given a web search query, retrieve relevant documents that answer the query",
            ],
        }

    def encode(
        self,
        sentences: Sequence[str],
        *,
        task_name: str,
        prompt_type: PromptType | None = None,
        **kwargs: Any,
    ) -> np.ndarray:
        """Encodes the given sentences using the encoder.

        Args:
            sentences: The sentences to encode.
            task_name: The name of the task.
            prompt_type: The prompt type to use.
            **kwargs: Additional arguments to pass to the encoder.

        Returns:
            The encoded sentences.
        """
        show_progress_bar = kwargs.get("show_progress_bar", True)
        if prompt_type == PromptType.query:
            # TODO:
            # 1) Implement the instruction task definition and
            #    remove the hard-coded task definition.
            # 2) Add option to not using any Instruction formatting.
            def format_query_inst(query: str) -> str:
                task_def = random.choice(self._task_name_2_inst_task_defs["msmarco"])
                query = f"Instruct: {task_def} \n Query: {query}"
                return query

            if self.use_query_inst_format:
                sentences = [format_query_inst(query) for query in sentences]
        # if prompt_type == PromptType.passage, encode the sentence directly as is

        lst_tokens = []
        for s in tqdm.tqdm(
            sentences,
            desc="Tokenize Sentences",
            disable=not show_progress_bar,
        ):
            t = torch.tensor(
                self.tokenizer.encode(s, add_bos=True, add_eos=True),
                dtype=torch.long,
                device=self._device,
            )
            lst_tokens.append(t)

        # Batching with Masking
        # TODO(pls331): Bi-directional mask. Default is causal mask if mask is None.
        batch_size = kwargs.get("batch_size", 1)
        lst_batch_tokens = []
        lst_batch_seqlen = []
        if batch_size > 1:
            for start in tqdm.tqdm(
                range(0, len(lst_tokens), batch_size),
                desc=f"Preprocess for Batching ({batch_size=})",
                disable=not show_progress_bar,
            ):
                end = min(start + batch_size, len(lst_tokens))
                batch_tokens: List[torch.Tensor] = lst_tokens[start:end]
                # batching_masks = torch.zeros(max_seqlen, dtype=torch.bool, device=self._device)
                batch_seqlen = torch.tensor( # B
                    [t.size(0) for t in batch_tokens], dtype=torch.int
                )
                max_seqlen = torch.max(batch_seqlen).item()
                batch_tokens_padded = torch.zeros( # [BS, max_seqlen]
                    (end - start, max_seqlen), dtype=torch.long, device=self._device
                )
                for i, t in enumerate(batch_tokens):
                    batch_tokens_padded[i, : t.size(0)] = t

                lst_batch_tokens.append(batch_tokens_padded)
                lst_batch_seqlen.append(batch_seqlen)
        else:
            # lst_batching_masks = [None] * len(lst_tokens)
            lst_batch_tokens = [t.unsqueeze(0) for t in lst_tokens]

        with torch.inference_mode():
            embeddings = []
            for i in tqdm.tqdm(
                range(len(lst_batch_tokens)), # num_batch x [B, max_seqlen(i)]
                desc="Encoding Embeddings",
                disable=not show_progress_bar,
            ):
                tokens = lst_batch_tokens[i]
                batch_seqlen = lst_batch_seqlen[i] if lst_batch_seqlen else None # [B]
                emb = self.model(tokens, batch_seqlen=batch_seqlen).detach().float().cpu().squeeze(0).numpy()
                embeddings.append(emb)
        res = np.concatenate(embeddings, axis=0)
        assert res.shape[0] == len(sentences)
        return res


class EvalRecipe:
    """
    Recipe for evaluating an embedding model on benchmarks:
    - MTEB: https://github.com/embeddings-benchmark/mteb

    This works for text-only text embedding model.

    This *does not* currently support the following features:
        - torch.compile
        - quantization through torchao
        - multi-GPU inference
    """

    def __init__(self, cfg: DictConfig) -> None:
        self._device: torch.device = utils.get_device(device=cfg.device)
        self._dtype: torch.dtype = training.get_dtype(
            dtype=cfg.dtype, device=self._device
        )
        self._logger: Any = utils.get_logger(cfg.log_level)
        self._embedding_model: Optional[torch.nn.Module] = None
        self.model: Optional[MTEBModelWrapper] = None
        self.tokenizer: Optional[ModelTokenizer] = None
        training.set_seed(seed=cfg.seed)

    def setup(self, cfg: DictConfig) -> None:
        """Setup the model and transforms."""
        # Load checkpointer and state_dict
        _checkpointer = config.instantiate(cfg.checkpointer)
        _ckpt_dict = _checkpointer.load_checkpoint()

        # Instantiate transforms
        self.tokenizer = config.instantiate(cfg.tokenizer)

        # Instantiate model
        with training.set_default_dtype(self._dtype), self._device:
            model = config.instantiate(cfg.model)
        assert hasattr(model, "decoder")
        model.decoder.load_state_dict(_ckpt_dict[training.MODEL_KEY])

        self._embedding_model = model
        self.model = MTEBModelWrapper(
            model=model,
            tokenizer=self.tokenizer,
            use_query_inst_format=True,
            device=self._device,
        )
        self._logger.info(f"Model was initialized with precision {self._dtype}.")

    def evaluate(self, cfg: DictConfig) -> None:
        """Evaluate the model using MTEB benchmark."""
        # Setup MTEB benchmark
        selected_tasks = []
        # for task_selector in cfg.eval.mteb.task_selectors:
        #     tasks = mteb.get_tasks(**task_selector)
        #     self._logger.info(f"Getting tasks for {task_selector}:\n{tasks}")
        #     selected_tasks.extend(tasks)
        task = mteb.get_task("HellaSwag")
        selected_tasks = [task]
        assert len(selected_tasks) > 0, "No tasks selected for evaluation."
        evaluation = mteb.MTEB(tasks=selected_tasks)
        evaluation.run(
            model=self.model, 
            output_dir=cfg.eval.output_dir, 
            encode_kwargs={
                "batch_size": cfg.eval.mteb.batch_size,
            },
        )


@config.parse
def main(cfg: DictConfig) -> None:
    config.log_config(recipe_name="EvalRecipe", cfg=cfg)
    recipe = EvalRecipe(cfg=cfg)
    recipe.setup(cfg=cfg)
    recipe.evaluate(cfg=cfg)


if __name__ == "__main__":
    sys.exit(main())
