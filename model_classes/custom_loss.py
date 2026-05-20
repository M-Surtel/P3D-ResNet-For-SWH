import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import _reduction as _Reduction, functional as F

import warnings
from typing import Optional

def mape_loss(
    input: Tensor,
    target: Tensor,
    size_average: Optional[bool] = None,
    reduce: Optional[bool] = None,
    reduction: str = "mean",
    weight: Optional[Tensor] = None,
) -> Tensor:
    if not (target.size() == input.size()):
        warnings.warn(
            f"Using a target size ({target.size()}) that is different to the input size ({input.size()}). "
            "This will likely lead to incorrect results due to broadcasting. "
            "Please ensure they have the same size.",
            stacklevel=2,
        )

    if size_average is not None or reduce is not None:
        reduction = _Reduction.legacy_get_string(size_average, reduce)

    expanded_input, expanded_target = torch.broadcast_tensors(input, target)
    abs_error_percentage = torch.abs(expanded_input - expanded_target) / expanded_target

    if weight is not None:
        if weight.size() != input.size():
            raise ValueError("Weights and input must have the same size.")
        abs_error_percentage = abs_error_percentage * weight

    if reduction == "none":
        return abs_error_percentage
    elif reduction == "sum":
        return torch.sum(abs_error_percentage)
    elif reduction == "mean":
        if weight is not None:
            return torch.sum(abs_error_percentage) / torch.sum(weight)
        else:
            return torch.mean(abs_error_percentage)
    else:
        raise ValueError(
            f"Invalid reduction mode: {reduction}. Expected one of 'none', 'mean', 'sum'."
        )


class MSExMAPELoss(nn.Module):
    reduction: str
    __constants__ = ["reduction", "weight"]
    def __init__(self, size_average=None, reduce=None, reduction: str = "mean", weight: float = 1) -> None:
        super().__init__()
        if size_average is not None or reduce is not None:
            self.reduction: str = _Reduction.legacy_get_string(size_average, reduce)
        else:
            self.reduction = reduction
        self.weight = weight


    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        return F.mse_loss(input, target, reduction=self.reduction) + self.weight * mape_loss(input, target, reduction=self.reduction)
