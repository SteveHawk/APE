import os
import torch
from typing import Tuple


def save_cp(Params: object, acc: float, name: str) -> None:
    checkpoint = {
        "epoch": Params.epoch,
        "model": Params.model,
        "optimizer": Params.opt.state_dict(),
        "scheduler": Params.scheduler.state_dict(),
        "acc": acc,
        "max_acc": Params.max_acc,
    }
    path = os.path.join(Params.model_path, name)
    torch.save(checkpoint, path)


def load_cp(model: object, opt: object, scheduler: object, model_path: str, name: str, dev: object) \
                                                    -> Tuple[int, object, object, object, float]:
    path = os.path.join(model_path, name)
    assert os.path.isfile(path)
    checkpoint = torch.load(path, map_location=dev)

    start_epoch = checkpoint["epoch"] + 1
    model = checkpoint["model"]
    opt.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    acc = checkpoint["acc"]
    max_acc = checkpoint["max_acc"]

    print(f"Loading checkpoint starting at epoch={start_epoch}, acc={acc}, max_acc={max_acc}.")
    return start_epoch, model, opt, scheduler, max_acc
