from __future__ import annotations

from typing import Any

import torch


def to_list(x: Any, n: int) -> list[Any]:
    if isinstance(x, (list, tuple)):
        values = list(x)
    elif torch.is_tensor(x):
        flat = x.detach().cpu().view(-1)
        values = [flat[i].item() for i in range(flat.numel())]
    elif n == 1:
        values = [x]
    else:
        raise ValueError(f"Unable to convert x:{x} to list of length:{n}")

    if len(values) != n:
        raise ValueError(f"Expected {n} values, got {len(values)}")
    return values


def infer_batch_size(x: Any, *, key: str) -> int:
    if isinstance(x, (list, tuple)):
        return len(x)
    if torch.is_tensor(x):
        if x.ndim == 0:
            raise ValueError(f"Batch key '{key}' must not be scalar tensor.")
        return int(x.shape[0])
    raise ValueError(f"Unable to infer batch size from key '{key}' with value type {type(x)}")
