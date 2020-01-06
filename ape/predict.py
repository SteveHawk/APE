import torch
from torch import nn
from torch.tensor import Tensor

import os
import glob
import json
import argparse
import importlib
from PIL import Image  # type: ignore
from typing import List, Tuple, Dict, Union

import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from ape.utils.model_store import load_model
from ape.utils.load_data import preprocess_x
from ape.utils.configs import Configs, config_path_process


def predict(images: List[Tuple[Tensor, str]], model: nn.Sequential, num_classes: int) \
                                    -> List[Dict[str, Union[str, int, List[float]]]]:
    results = list()
    with torch.no_grad():
        for img, path in images:
            print(path, end="\r")

            outputs = model(img)
            result: Dict[str, Union[str, int, List[float]]] = dict()
            result["path"] = path

            # label index
            _, predicted = torch.max(outputs.data, 1)
            result["result"] = int(predicted.item())

            # possibilities
            poss = nn.functional.softmax(outputs, dim=1)
            poss_list = list()
            for i in range(num_classes):
                poss_list.append(poss.data[0][i].item())
            result["possibilities"] = poss_list

            results.append(result)
    return results


def prepare(configs: Configs) -> Tuple[nn.Sequential, torch.device]:
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

    return model, dev


def img_prepare(img_path: str, ext: str, configs: Configs, dev: torch.device) -> List[Tuple[Tensor, str]]:
    imgs = list()
    paths = glob.glob(os.path.join(img_path, f"*.{ext}"))
    for path in paths:
        img = Image.open(path)
        imgs.append((preprocess_x(img, dev, configs.img_size_x, configs.img_size_y,
            configs.ds_mean, configs.ds_std, configs.gray_scale), path))
    return imgs


def write_json(result: List[Dict[str, Union[str, int, List[float]]]], output_path: str) -> None:
    path = os.path.join(output_path, "output.json")
    with open(path, "w") as f:
        json.dump(result, f)
    print(f"Result saved at {path}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict pictures using trained models.")
    parser.add_argument("--config", dest="config_path", nargs=1, required=True, help="specify the config location")
    parser.add_argument("--path", dest="img_path", nargs=1, required=True, help="specify the image folder path")
    parser.add_argument("--ext", dest="ext", nargs=1, required=True, help="specify the image extension name")
    parser.add_argument("--output", dest="output_path", nargs=1, required=True, help="specify the output path")
    args = parser.parse_args()

    config_path = args.config_path[0]
    img_path = args.img_path[0]
    ext = args.ext[0]
    output_path = args.output_path[0]

    assert os.path.isfile(config_path)
    assert os.path.exists(img_path)
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    assert os.path.exists(output_path)

    configs: Configs = importlib.import_module(config_path_process(config_path)).Configs  # type: ignore

    model, dev = prepare(configs)
    imgs = img_prepare(img_path, ext, configs, dev)
    result = predict(imgs, model, configs.num_classes)
    write_json(result, output_path)
