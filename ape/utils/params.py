import torch
from typing import Callable, List

from .load_data import WrappedDataLoader


class Params:
    model: torch.nn.Sequential
    num_epochs: int
    start_epoch: int
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    opt: torch.optim.SGD
    scheduler: torch.optim.lr_scheduler.ExponentialLR
    train_dl: WrappedDataLoader
    valid_dl: WrappedDataLoader
    max_acc: float
    target_acc: float
    verbose: List[bool]
    model_path: str
    writer: torch.utils.tensorboard.SummaryWriter
    epoch: int
