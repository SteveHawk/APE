import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import transforms
from torchvision.utils import make_grid
from torchvision.datasets import ImageFolder
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

import os
import argparse
import importlib
import numpy as np
from PIL import Image
from typing import Tuple


class WrappedDataLoader:
    def __init__(self, dl, func):
        self.dl = dl
        self.func = func

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class Params:
    pass


def transform(transform_img_size_x, transform_img_size_y, ds_mean, ds_std):
    def _transform(img):
        img = img.convert("L")
        size = img.size
        if size[0] > size[1]:
            img = img.rotate(90)
        img = img.resize((transform_img_size_y, transform_img_size_x), Image.ANTIALIAS)

        transform_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean = [ds_mean], std = [ds_std])
            ]
        )
        return transform_tensor(img)
    return _transform


def preprocess(transform_img_size_x, transform_img_size_y, dev):
    def _preprocess(x, y):
        return x.view(-1, 1, transform_img_size_x, transform_img_size_y).to(dev), y.to(dev)
    return _preprocess


def load_data(data_path, transform, bs, transform_img_size_x, transform_img_size_y, dev, num_workers, ds_mean, ds_std) -> Tuple[WrappedDataLoader]:
    full_ds = ImageFolder(data_path, transform=transform(transform_img_size_x, transform_img_size_y, ds_mean, ds_std))
    print("ImageFolder Done.")
    print(full_ds.class_to_idx)

    train_size = int(0.8 * len(full_ds))
    valid_size = len(full_ds) - train_size
    train_ds, valid_ds = random_split(full_ds, [train_size, valid_size])
    print("Split Done.")

    # Change num_workers to enable multi-thread / multi-process
    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers)
    valid_dl = DataLoader(valid_ds, batch_size=bs, num_workers=num_workers)
    print("DataLoader Done.")

    train_dl = WrappedDataLoader(train_dl, preprocess(transform_img_size_x, transform_img_size_y, dev))
    valid_dl = WrappedDataLoader(valid_dl, preprocess(transform_img_size_x, transform_img_size_y, dev))
    print("WrappedDataLoader Done.")

    return train_dl, valid_dl


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def save_cp(model_params, epoch, acc, name) -> None:
    checkpoint = {
        "epoch": epoch,
        "model": model_params.model,
        "optimizer": model_params.opt.state_dict(),
        "scheduler": model_params.scheduler.state_dict(),
        "acc": acc,
        "max_acc": model_params.max_acc,
    }
    path = os.path.join(model_params.model_path, name)
    torch.save(checkpoint, path)


def load_cp(model, opt, scheduler, model_path, name, dev):
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


def cal_loss(model, epoch, loss_func, dl, verbose, writer, catagory) -> float:
    model.eval()
    if verbose:
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch(model, loss_func, xb, yb) for xb, yb in dl]
            )
        loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        writer.add_scalar(catagory, loss, epoch)
    else:
        loss = "off"
    return loss


def cal_acc(model, epoch, dl, verbose, writer, catagory) -> float:
    model.eval()
    if verbose:
        correct = 0
        total = 0
        with torch.no_grad():
            for data in dl:
                images, labels = data
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        acc = 100 * correct / total
        writer.add_scalar(catagory, acc, epoch)
    else:
        acc = "off"
    return acc


def training_info(model_params: object, epoch: int) -> Tuple[float]:
    model = model_params.model
    loss_func = model_params.loss_func
    train_dl = model_params.train_dl
    valid_dl = model_params.valid_dl
    verbose = model_params.verbose
    writer = model_params.writer

    train_loss = cal_loss(model, epoch, loss_func, train_dl, verbose[0], writer, "Loss/train")
    valid_loss = cal_loss(model, epoch, loss_func, valid_dl, verbose[1], writer, "Loss/validation")
    train_acc = cal_acc(model, epoch, train_dl, verbose[2], writer, "Accuracy/train")
    valid_acc = cal_acc(model, epoch, valid_dl, verbose[3], writer, "Accuracy/validation")

    return train_loss, valid_loss, train_acc, valid_acc


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
        start_epoch, model, opt, scheduler, max_acc = load_cp(model, opt, scheduler, Configs.model_path, Configs.resume_model_name, dev)
    else:
        start_epoch = 0
        max_acc = 0
    model.to(dev)
    loss_func = F.cross_entropy

    # Model info
    print(f"Model:\n{model}")
    print(f"Optimizer:\n{opt}")

    # Load data
    train_dl, valid_dl = load_data(Configs.data_path, transform, Configs.bs, Configs.transform_img_size_x, Configs.transform_img_size_y, dev, Configs.num_workers, Configs.ds_mean, Configs.ds_std)

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


def train() -> None:
    # Info header
    print("epoch | train_loss | valid_loss | train_acc | valid_acc")

    # Start training epochs
    for epoch in range(Params.start_epoch, Params.num_epochs):
        # Training mode
        Params.model.train()
        counter = 0
        for xb, yb in Params.train_dl:
            loss_batch(Params.model, Params.loss_func, xb, yb, Params.opt)
            # Epoch progress bar
            if Params.verbose[4]:
                counter += 1
                progress = round(100 * counter / len(Params.train_dl), 2)
                print("Epoch progress: |" + "*"*int(progress/5) + "_"*int(20-progress/5) + f"| {progress}%", end="\r")
        Params.scheduler.step()

        # Training info calculation
        train_loss, valid_loss, train_acc, valid_acc = training_info(Params, epoch)
        print(f"{epoch} | {train_loss} | {valid_loss} | {train_acc} | {valid_acc}")

        # Save models
        if Params.verbose[3]:
            # Save best record
            if valid_acc > Params.max_acc:
                Params.max_acc = valid_acc
                save_cp(Params, epoch, valid_acc, f"model_checkpoint_max_acc.pth.tar")
            # Save target model
            if valid_acc > Params.target_acc:
                save_cp(Params, epoch, valid_acc, f"target_acc_{epoch}.pth.tar")
                print("Accuracy target reached, model saved. Training stopped.")
                Params.writer.close()
                return

        # Save checkpoint
        save_cp(Params, epoch, valid_acc, f"model_checkpoint_{epoch}.pth.tar")
        print(f"Checkpoint of epoch={epoch} saved.")


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
