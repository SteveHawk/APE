import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from time import time
from tools.train import View, WrappedDataLoader, transform, preprocess, config_path
import argparse
import importlib


def load_cp(model_path, name, dev):
    path = os.path.join(model_path, name)
    assert os.path.isfile(path)
    checkpoint = torch.load(path, map_location=dev)
    epoch = checkpoint["epoch"]
    model = checkpoint["model"]
    acc = checkpoint["acc"]
    max_acc = checkpoint["max_acc"]
    print(f"Loading checkpoint of epoch={epoch}, acc={acc}, max_acc={max_acc}.")
    return model


def test(Configs: object) -> None:
    # Device settings
    if Configs.dev_num is None:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device(f"cuda:{Configs.dev_num}")
        torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")

    # Model loading
    model = load_cp(Configs.model_path, Configs.test_model_name, dev).to(dev)
    print(model)
    print("Model Loading Done.")
    
    # Data loading
    test_ds = ImageFolder(Configs.test_data_path, transform=transform(Configs.transform_img_size_x, Configs.transform_img_size_y, Configs.ds_mean, Configs.ds_std))
    print("ImageFolder Done.")
    print(test_ds.class_to_idx)
    test_dl = WrappedDataLoader(DataLoader(test_ds, batch_size=Configs.bs, num_workers=Configs.num_workers), preprocess(Configs.transform_img_size_x, Configs.transform_img_size_y, dev))
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


def config_path_process(path: str) -> str:
    path = path.lstrip("./\\")
    path = path.rstrip("py")
    path = path.rstrip(".")
    path = path.replace("/", ".")
    path = path.replace("\\", ".")
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test the model's accuracy and performance.")
    parser.add_argument("--config", dest="config_path", nargs=1, required=True, help="specify the config location")
    args = parser.parse_args()

    config_path = args.config_path[0]
    assert os.path.isfile(config_path)
    configs = importlib.import_module(config_path_process(config_path))

    test(configs.Configs)
