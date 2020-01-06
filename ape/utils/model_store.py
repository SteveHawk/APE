import os
import torch
from typing import Tuple
from torch.optim import lr_scheduler, SGD

from ape.utils.params import Params


def save_cp(acc: float, name: str) -> None:
    checkpoint = {
        "epoch": Params.epoch,
        "model": Params.model.state_dict(),
        "optimizer": Params.opt.state_dict(),
        "scheduler": Params.scheduler.state_dict(),
        "acc": acc,
        "max_acc": Params.max_acc,
    }
    path = os.path.join(Params.model_path, name)
    torch.save(checkpoint, path)


def load_cp(model: torch.nn.Sequential, opt: SGD, scheduler: lr_scheduler.ExponentialLR, model_path: str,
                                    name: str, dev: torch.device) -> Tuple[int, torch.nn.Sequential, SGD,
                                    lr_scheduler.ExponentialLR, float]:
    path = os.path.join(model_path, name)
    assert os.path.isfile(path)
    checkpoint = torch.load(path, map_location=dev)

    start_epoch = checkpoint["epoch"]
    model.load_state_dict(checkpoint["model"])
    opt.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    acc = checkpoint["acc"]
    max_acc = checkpoint["max_acc"]

    print(f"Loading checkpoint starting at epoch={start_epoch}, acc={acc}, max_acc={max_acc}.")
    return start_epoch, model, opt, scheduler, max_acc


def load_model(model: torch.nn.Sequential, model_path: str, name: str, dev: torch.device) -> torch.nn.Sequential:
    path = os.path.join(model_path, name)
    assert os.path.isfile(path)
    checkpoint = torch.load(path, map_location=dev)

    model.load_state_dict(checkpoint["model"])
    model.to(dev)
    epoch = checkpoint["epoch"]
    acc = checkpoint["acc"]
    max_acc = checkpoint["max_acc"]

    print(f"Loading model of epoch={epoch}, acc={acc}, max_acc={max_acc}.")
    return model
