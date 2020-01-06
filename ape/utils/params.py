import torch
from torch.utils.tensorboard import SummaryWriter
from typing import Callable, List

from ape.utils.load_data import WrappedDataLoader


class Params:
    model: torch.nn.Sequential
    max_epochs: int
    start_epoch: int
    loss_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    opt: torch.optim.SGD
    scheduler: torch.optim.lr_scheduler.ExponentialLR
    train_dl: WrappedDataLoader
    valid_dl: WrappedDataLoader
    max_acc: float
    target_acc: float
    log_step: int
    verbose: List[bool]
    model_path: str
    writer: SummaryWriter
    epoch: int
    steps: int
