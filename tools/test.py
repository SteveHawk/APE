import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
import os
from time import time
from train import Params, View, WrappedDataLoader, transform, preprocess, config_path
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


def test(params):
    # Device settings
    if not params.dev_num:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device(f"cuda:{params.dev_num}")
        torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")

    # Model loading
    model = load_cp(params.model_path, params.test_model_name, dev).to(dev)
    print(model)
    print("Model Loading Done.")
    
    # Data loading
    test_ds = ImageFolder(params.test_data_path, transform=transform(params.transform_img_size_x, params.transform_img_size_y, params.ds_mean, params.ds_std))
    print("ImageFolder Done.")
    print(test_ds.class_to_idx)
    test_dl = WrappedDataLoader(DataLoader(test_ds, batch_size=params.bs, num_workers=params.num_workers), preprocess(params.transform_img_size_x, params.transform_img_size_y, dev))
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
    params = importlib.import_module(config_path(args.config_path[0]))
    test(params.params)
