# Import PyTorch lazily so data/evaluation code can be used without it.
from __future__ import annotations


def torch_nn():
    # Return (torch, torch.nn); raise a helpful error when PyTorch is absent.
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise ImportError("PyTorch is required for models; install it from "
                          "https://pytorch.org/get-started/locally/") from exc
    return torch, nn
