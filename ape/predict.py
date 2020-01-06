import torch
from torch import nn
from torch.tensor import Tensor

import os
import glob
import json
import argparse
import importlib
import multiprocessing
from PIL import Image  # type: ignore
from joblib import Parallel, delayed  # type: ignore
from typing import List, Tuple, Dict, Union

import sys
sys.path.insert(1, os.path.join(sys.path[0], ".."))

from ape.utils.model_store import load_model
from ape.utils.load_data import preprocess_x
from ape.utils.configs import Configs, config_path_process


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


def predict(path: str, configs: Configs, dev: torch.device, model: nn.Sequential,
                results: List[Dict[str, Union[str, int, List[float]]]]) -> None:
    print(path, end="\r")

    img_raw = Image.open(path)
    img = preprocess_x(img_raw, dev, configs.img_size_x, configs.img_size_y,
        configs.ds_mean, configs.ds_std, configs.gray_scale)

    with torch.no_grad():
        outputs = model(img)
        result: Dict[str, Union[str, int, List[float]]] = dict()
        result["path"] = path

        # label index
        _, predicted = torch.max(outputs.data, 1)
        result["result"] = int(predicted.item())

        # possibilities
        poss = nn.functional.softmax(outputs, dim=1)
        result["possibilities"] = poss.data[0].tolist()

    results.append(result)


def img_predict(img_path: str, ext: str, configs: Configs, dev: torch.device, model: nn.Sequential) \
                                                    -> List[Dict[str, Union[str, int, List[float]]]]:
    paths = glob.glob(os.path.join(img_path, f"*.{ext}"))

    n_process = multiprocessing.cpu_count()
    print(f"Using {n_process} threads.")

    results: List[Dict[str, Union[str, int, List[float]]]] = list()
    Parallel(n_jobs=n_process, require="sharedmem")(
        delayed(predict)(path, configs, dev, model, results) for path in paths
    )

    return results


def write_json(results: List[Dict[str, Union[str, int, List[float]]]], output_path: str) -> None:
    path = os.path.join(output_path, "output.json")
    with open(path, "w") as f:
        json.dump(results, f)
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
    results = img_predict(img_path, ext, configs, dev, model)
    write_json(results, output_path)
