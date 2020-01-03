import torch
import numpy as np  # type: ignore
from typing import Tuple, Union

from .params import Params
from .load_data import WrappedDataLoader


def cal_loss(dl: WrappedDataLoader, verbose: bool, catagory: str) -> Union[float, str]:
    Params.model.eval()
    if verbose:
        with torch.no_grad():
            losses, nums = zip(
                *[(Params.loss_func(Params.model(xb), yb).item(), len(xb)) for xb, yb in dl]
            )
        loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        Params.writer.add_scalar(catagory, loss, Params.epoch)
        return loss
    else:
        return "off"


def cal_acc(dl: WrappedDataLoader, verbose: bool, catagory: str) -> Union[float, str]:
    Params.model.eval()
    if verbose:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dl:
                images, labels = data
                outputs = Params.model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += int((predicted == labels).sum().item())
        acc = 100 * correct / total
        Params.writer.add_scalar(catagory, acc, Params.epoch)
        return acc
    else:
        return "off"


def training_info() -> Tuple[Union[float, str], Union[float, str],
                            Union[float, str], Union[float, str]]:
    train_dl = Params.train_dl
    valid_dl = Params.valid_dl
    verbose = Params.verbose

    train_loss = cal_loss(train_dl, verbose[0], "Loss/train")
    valid_loss = cal_loss(valid_dl, verbose[1], "Loss/validation")
    train_acc = cal_acc(train_dl, verbose[2], "Accuracy/train")
    valid_acc = cal_acc(valid_dl, verbose[3], "Accuracy/validation")

    return train_loss, valid_loss, train_acc, valid_acc
