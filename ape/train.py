import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import importlib
from typing import Tuple, List, Callable

import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from ape.utils.params import Params
from ape.utils.configs import Configs, config_path_process
from ape.utils import info_cal, load_data, model_store


def save_model(valid_acc: float) -> bool:
    # Save models
    if Params.verbose[3]:
        # Save best record
        if valid_acc > Params.max_acc:
            Params.max_acc = valid_acc
            model_store.save_cp(valid_acc, f"model_cp_max_acc.pth.tar")
        # Save target model
        if valid_acc > Params.target_acc:
            model_store.save_cp(valid_acc, f"target_acc_{Params.steps}.pth.tar")
            print("Accuracy target reached, model saved. Training stopped.")
            Params.writer.close()
            return True

    # Save checkpoint
    model_store.save_cp(valid_acc, f"model_cp_{Params.steps}.pth.tar")
    print(f"Checkpoint of epoch={Params.epoch}, step={Params.steps} saved.")
    return False


def progress_bar(progress: float, steps: int, epoch: int) -> None:
    print(f"Epoch {epoch} progress: |", end="")
    print("*" * int(progress / 5), end="")
    print("_" * int(20 - progress / 5), end="")
    print(f"| {round(progress, 2)}%, step {steps}    ", end="\r")


def loss_batch(xb: torch.Tensor, yb: torch.Tensor) -> None:
    Params.model.train()
    loss = Params.loss_func(Params.model(xb), yb)

    loss.backward()
    Params.opt.step()
    Params.opt.zero_grad()


def train() -> None:
    print("step | train_loss | valid_loss | train_acc | valid_acc")
    Params.steps = 0
    for epoch in range(Params.start_epoch, Params.max_epochs):
        Params.epoch = epoch
        counter = 0

        for xb, yb in Params.train_dl:
            loss_batch(xb, yb)

            Params.steps += 1
            if Params.steps % Params.log_step == 0:
                train_loss, valid_loss, train_acc, valid_acc = info_cal.training_info()
                print(f"{Params.steps} | {train_loss} | {valid_loss} | {train_acc} | {valid_acc}    ")
                if Params.verbose[3] and isinstance(valid_acc, float) and save_model(valid_acc):
                    return

            if Params.verbose[4]:
                counter += 1
                progress_bar(100 * counter / len(Params.train_dl), Params.steps, epoch)
        Params.scheduler.step(epoch)


def prepare(configs: Configs) -> None:
    # Device settings
    if configs.dev_num is None:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device(f"cuda:{configs.dev_num}")
        # torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")

    # Load model
    print(f"Resume: {configs.resume}")
    model = configs.model
    model.to(dev)
    opt = optim.SGD(model.parameters(), lr=configs.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9, last_epoch=-1)
    if configs.resume:
        start_epoch, model, opt, scheduler, max_acc = model_store.load_cp(model, opt, scheduler,
            configs.model_path, configs.resume_model_name, dev)
    else:
        start_epoch = 0
        max_acc = 0
    loss_func = F.cross_entropy

    # Model info
    print(f"Model:\n{model}")
    print(f"Optimizer:\n{opt}")

    # Load data
    train_dl, valid_dl = load_data.load_data(configs, dev)

    # Create model_path folder
    if not os.path.exists(configs.model_path):
        os.makedirs(configs.model_path)
    assert os.path.exists(configs.model_path)

    # For Tensorboard
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(flush_secs=30)

    # Load the params
    Params.model = model
    Params.max_epochs = configs.max_epochs
    Params.start_epoch = start_epoch
    Params.loss_func = loss_func
    Params.opt = opt
    Params.scheduler = scheduler
    Params.train_dl = train_dl
    Params.valid_dl = valid_dl
    Params.max_acc = max_acc
    Params.target_acc = configs.target_acc
    Params.log_step = configs.log_step
    Params.verbose = configs.verbose
    Params.model_path = configs.model_path
    Params.writer = writer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APE, A simPle image classification framEwork")
    parser.add_argument("--config", dest="config_path", nargs=1, required=True, help="specify the config location")
    args = parser.parse_args()

    config_path = args.config_path[0]
    assert os.path.isfile(config_path)

    configs = importlib.import_module(config_path_process(config_path)).Configs  # type: ignore

    prepare(configs)
    train()
