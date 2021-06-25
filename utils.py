from typing import Callable, Any, Optional, List

import torch


def make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def calc_acc(preds: torch.Tensor, labels: torch.Tensor):
    _, pred_max = torch.max(preds, 1)
    acc = torch.sum(pred_max == labels.data, dtype=torch.float64) / len(preds)
    return acc