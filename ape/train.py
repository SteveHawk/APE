import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import os
import argparse
import importlib
from typing import Tuple, Union

from utils import info_cal, load_data, model_store


class Params:
    pass


def loss_batch(Params: object, xb: object, yb: object) -> Tuple[object, int]:
    loss = Params.loss_func(Params.model(xb), yb)

    loss.backward()
    Params.opt.step()
    Params.opt.zero_grad()

    return loss.item(), len(xb)


def train() -> None:
    # Info header
    print("epoch | train_loss | valid_loss | train_acc | valid_acc")

    # Start training epochs
    for epoch in range(Params.start_epoch, Params.num_epochs):
        # Add epoch info in Params
        Params.epoch = epoch

        # Training mode
        Params.model.train()
        counter = 0
        for xb, yb in Params.train_dl:
            loss_batch(Params, xb, yb)
            # Epoch progress bar
            if Params.verbose[4]:
                counter += 1
                progress = round(100 * counter / len(Params.train_dl), 2)
                print("Epoch progress: |" + "*"*int(progress/5) + "_"*int(20-progress/5) + f"| {progress}%", end="\r")
        Params.scheduler.step()

        # Training info calculation
        train_loss, valid_loss, train_acc, valid_acc = info_cal.training_info(Params)
        print(f"{epoch} | {train_loss} | {valid_loss} | {train_acc} | {valid_acc}")

        # Save models
        if Params.verbose[3]:
            # Save best record
            if valid_acc > Params.max_acc:
                Params.max_acc = valid_acc
                model_store.save_cp(valid_acc, f"model_checkpoint_max_acc.pth.tar")
            # Save target model
            if valid_acc > Params.target_acc:
                model_store.save_cp(valid_acc, f"target_acc_{epoch}.pth.tar")
                print("Accuracy target reached, model saved. Training stopped.")
                Params.writer.close()
                return

        # Save checkpoint
        model_store.save_cp(valid_acc, f"model_checkpoint_{epoch}.pth.tar")
        print(f"Checkpoint of epoch={epoch} saved.")


def prepare(Configs: object) -> None:
    # Device settings
    if Configs.dev_num is None:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device(f"cuda:{Configs.dev_num}")
        torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")

    # Load model
    print(f"Resume: {Configs.resume}")
    model = Configs.model
    opt = optim.SGD(model.parameters(), lr=Configs.lr, momentum=0.9)
    scheduler = optim.lr_scheduler.ExponentialLR(opt, gamma=0.9, last_epoch=-1)
    if Configs.resume:
        start_epoch, model, opt, scheduler, max_acc = model_store.load_cp(model, opt, scheduler,
        Configs.model_path, Configs.resume_model_name, dev)
    else:
        start_epoch = 0
        max_acc = 0
    model.to(dev)
    loss_func = F.cross_entropy

    # Model info
    print(f"Model:\n{model}")
    print(f"Optimizer:\n{opt}")

    # Load data
    train_dl, valid_dl = load_data.load_data(Configs.data_path, Configs.bs, Configs.transform_img_size_x,
    Configs.transform_img_size_y, dev, Configs.num_workers, Configs.ds_mean, Configs.ds_std)

    # Create model_path folder
    if not os.path.exists(Configs.model_path):
        os.makedirs(Configs.model_path)
    assert os.path.exists(Configs.model_path)

    # For Tensorboard
    # Writer will output to ./runs/ directory by default
    writer = SummaryWriter(flush_secs=30)

    # Load the params
    Params.model = model
    Params.num_epochs = Configs.num_epochs
    Params.start_epoch = start_epoch
    Params.loss_func = loss_func
    Params.opt = opt
    Params.scheduler = scheduler
    Params.train_dl = train_dl
    Params.valid_dl = valid_dl
    Params.max_acc = max_acc
    Params.target_acc = Configs.target_acc
    Params.verbose = Configs.verbose
    Params.model_path = Configs.model_path
    Params.writer = writer


def config_path_process(path: str) -> str:
    path = path.lstrip("./\\")
    path = path.rstrip("py")
    path = path.rstrip(".")
    path = path.replace("/", ".")
    path = path.replace("\\", ".")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="APE, A simPle image classification framEwork")
    parser.add_argument("--config", dest="config_path", nargs=1, required=True, help="specify the config location")
    args = parser.parse_args()

    config_path = args.config_path[0]
    assert os.path.isfile(config_path)

    import sys
    sys.path.insert(1, os.path.join(sys.path[0], ".."))
    configs = importlib.import_module(config_path_process(config_path))

    prepare(configs.Configs)
    train()
