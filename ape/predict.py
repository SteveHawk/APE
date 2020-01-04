import torch
from torch import nn
from torch.tensor import Tensor

import os
import glob
import json
import argparse
import importlib
from PIL import Image
from typing import List, Tuple, Dict

from ape.utils.model_store import load_model
from ape.utils.load_data import preprocess_x
from ape.utils.configs import Configs, config_path_process


def predict(images: List[Tensor], model: nn.Sequential) -> List[Tuple[float, ...]]:
    labels: List[Tuple[float, ...]] = list()
    with torch.no_grad():
        for img in images:
            outputs = model(img)

            # # Return label index
            # _, predicted = torch.max(outputs.data, 1)
            # labels.append(predicted.item())

            # Return possibilities
            outputs = nn.functional.softmax(outputs, dim=1)
            # TODO: Multi class
            labels.append((outputs.data[0][0].item(), outputs.data[0][1].item()))

    return labels


def prepare(configs: Configs, img_path: str) -> Tuple[List[Tensor], nn.Sequential]:
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
    model = load_model(configs.model, configs.model_path, configs.prediction_model_name, dev)
    model.eval()
    print(model)
    print("Model Loading Done.")

    # Prepare imgs
    imgs = list()
    for i in range(1, 11):
        img = Image.open(f"./ds/img ({i}).png")
        imgs.append(preprocess_x(img, dev, configs.img_size_x, configs.img_size_y,
            configs.ds_mean, configs.ds_std, configs.gray_scale))

    return imgs, model


def write_json(result: List[Tuple[float, ...]], output_path: str) -> None:
    output: Dict[str, str] = dict()
    with open(output_path, "w") as f:
        json.dump(output, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pictures using trained models.")
    parser.add_argument("--config", dest="config_path", nargs=1, required=True, help="specify the config location")
    parser.add_argument("--path", dest="img_path", nargs=1, required=True, help="specify the image folder path")
    parser.add_argument("--output", dest="output_path", nargs=1, required=True, help="specify the output path")
    args = parser.parse_args()

    config_path = args.config_path[0]
    img_path = args.config_path[1]
    output_path = args.config_path[2]

    assert os.path.isfile(config_path)
    assert os.path.exists(img_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.exists(output_path)

    configs = importlib.import_module(config_path_process(config_path)).Configs  # type: ignore

    result = predict(*prepare(configs, img_path))
    write_json(result, output_path)
