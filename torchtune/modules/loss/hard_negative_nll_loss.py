from typing import Optional
import torch
from torch import nn, Tensor

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.dim() in (1, 2)
    a_ = torch.nn.functional.normalize(a, dim=1, p=2)
    b_ = torch.nn.functional.normalize(b, dim=1, p=2)
    return (a_ * b_).sum(dim=1)

def cosine_distance():
    return lambda a, b: 1 - cosine_similarity(a, b)

def cosine_similarity_a2a(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.dim() == 2
    a_ = torch.nn.functional.normalize(a, dim=1, p=2)
    b_ = torch.nn.functional.normalize(b, dim=1, p=2)
    return torch.mm(a_, b_.t())

class HardNegativeNLLLoss:
    """
    Supports both pair and triplet loss.
    TODO (pls331): support distributed training.
    """
    def __init__(
        self,
        scale: float = 20.0,
        similarity_fn=cosine_similarity_a2a,
    ):
        self._scale = scale
        self.similarity_fn = similarity_fn
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def __call__(
        self,
        embs_query: Tensor,
        embs_pos: Tensor,
        embs_neg: Optional[Tensor] = None,
    ):
        if embs_neg is None:
            embs_neg = embs_pos[:0, :]

        embs_doc = torch.cat([embs_pos, embs_neg], dim=0)
        scores = self.similarity_fn(embs_query, embs_doc) * self._scale
        labels = torch.tensor(
            range(len(scores)), dtype=torch.long, device=scores.device
        )
        # import pdb; pdb.set_trace()
        loss = self.cross_entropy_loss(scores, labels)
        return loss