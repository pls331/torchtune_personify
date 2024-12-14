import torch

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.shape == b.shape and a.dim() in (1, 2)
    a_ = torch.nn.functional.normalize(a, dim=1, p=2)
    b_ = torch.nn.functional.normalize(b, dim=1, p=2)
    return (a_ * b_).sum(dim=1)

def cosine_distance():
    return lambda a, b: 1 - cosine_similarity(a, b)

# class HardNegativeNLLLoss: