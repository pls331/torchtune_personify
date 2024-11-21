import random

from typing import Any, Callable, Dict, List, Mapping, Optional, Union

from datasets import load_dataset
from torch.utils.data import Dataset
from torchtune.data._utils import truncate
from torchtune.datasets._packed import PackedDataset
from torchtune.modules.tokenizers import ModelTokenizer


class RetrievalTripletDataset(Dataset):
    """
    The triplet is formated following instruct template
    `q_inst = Instruct: {task_definition} \n Query: {query}`
    This is introduced from paper:
    1.Wang, L. et al. Improving Text Embeddings with Large Language Models. arXiv (2023) doi:10.48550/arxiv.2401.00368.
    """

    def __init__(
        self,
        tokenizer: ModelTokenizer,
        source: str,
        column_query: str = "query",
        column_positive: str = "positive",
        column_negative: str = "negative",
        inst_task_def: Optional[List[str]] = None,
        add_eos: bool = True,
        filter_fn: Optional[Callable] = None,
        **load_dataset_kwargs: Dict[str, Any],
    ) -> None:
        self._tokenizer = tokenizer
        self._data = load_dataset(source, **load_dataset_kwargs)
        self.add_eos = add_eos
        self._column_query, self._column_positive, self._column_negative = (
            column_query,
            column_positive,
            column_negative,
        )
        self._inst_task_def = inst_task_def

        if filter_fn is not None:
            self._data = self._data.filter(filter_fn)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._data[index]
        return self._prepare_sample(sample)

    def _prepare_sample(self, sample: Mapping[str, Any]) -> Dict[str, List[int]]:
        query, positive, negative = (
            sample[self._column_query],
            sample[self._column_positive],
            sample[self._column_negative],
        )

        if self._inst_task_def:
            if isinstance(self._inst_task_def, list):
                task_def = random.choice(self._inst_task_def)
            else:
                assert isinstance(self._inst_task_def, str), type(self._inst_task_def)
                task_def = self._inst_task_def
            query = f"Instruct: {task_def} \n Query: {query}"

        tokens_query = self._tokenizer.encode(
            text=query, add_bos=True, add_eos=self.add_eos
        )
        tokens_positive = self._tokenizer.encode(
            text=positive, add_bos=True, add_eos=self.add_eos
        )
        tokens_negative = self._tokenizer.encode(
            text=negative, add_bos=True, add_eos=self.add_eos
        )

        # Truncate if needed, but don't coerce EOS id
        if self._tokenizer.max_seq_len is not None:
            tokens_query_inst = truncate(tokens_query, self._tokenizer.max_seq_len - 1)
            tokens_positive = truncate(tokens_positive, self._tokenizer.max_seq_len - 1)
            tokens_negative = truncate(tokens_negative, self._tokenizer.max_seq_len - 1)

        return {
            self._column_query: tokens_query,
            self._column_positive: tokens_positive,
            self._column_negative: tokens_negative,
        }


def msmarco_dataset(
    tokenizer: ModelTokenizer,
    source: str = "sentence-transformers/msmarco-msmarco-distilbert-base-tas-b",
    subset: str = "triplet",
    max_seq_len: Optional[int] = None,
    packed: bool = False,  # TODO(pls331) : maybe not necessary to support packing?
    filter_fn: Optional[Callable] = None,
    split: str = "train",
    **load_dataset_kwargs: Dict[str, Any],
) -> Union[RetrievalTripletDataset]:
    ds = RetrievalTripletDataset(
        tokenizer=tokenizer,
        source=source,
        name=subset,
        column_query="query",
        column_positive="positive",
        column_negative="negative",
        inst_task_def=[
            "Given a web search query, retrieve relevant passages that answer the query",
            "Given a web search query, retrieve relevant documents that answer the query",
        ],
        split=split,
        add_eos=True,
    )
    return ds
