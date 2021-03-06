import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder  # type: ignore

import os
import argparse
import importlib
from time import time

import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from ape.utils.model_store import load_model
from ape.utils.configs import Configs, config_path_process
from ape.utils.load_data import WrappedDataLoader, preprocess, transform


def test(configs: Configs) -> None:
    # Device settings
    if configs.dev_num is None:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device(f"cuda:{configs.dev_num}")
        # torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")

    # Model loading
    model = load_model(configs.model, configs.model_path, configs.test_model_name, dev)
    print(model)
    print("Model Loading Done.")

    # Data loading
    test_ds = ImageFolder(configs.test_data_path, transform=transform(configs.img_size_x, configs.img_size_y,
        configs.ds_mean, configs.ds_std, configs.gray_scale))
    print("ImageFolder Done.")
    print(test_ds.class_to_idx)
    test_dl = WrappedDataLoader(DataLoader(test_ds, batch_size=configs.bs, num_workers=configs.num_workers), preprocess(configs.img_size_x, configs.img_size_y, dev, configs.gray_scale))
    print("DataLoader Done.")

    model.eval()
    start = time()

    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_dl:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    test_acc = 100 * correct / total

    end = time()
    print(f"Total time cost: {end - start}s. Avg time per pic: {(end - start) / total}s")

    print(f"Total: {total}, Correct: {correct}")
    print(f"Test Set Accuracy: {test_acc}%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model's accuracy and performance.")
    parser.add_argument("--config", dest="config_path", nargs=1, required=True, help="specify the config location")
    args = parser.parse_args()

    config_path = args.config_path[0]
    assert os.path.isfile(config_path)

    configs = importlib.import_module(config_path_process(config_path)).Configs  # type: ignore

    test(configs)
