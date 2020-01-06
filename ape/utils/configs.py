from torch import nn
from typing import List, Optional


class Configs:
    num_classes: int
    gray_scale: bool
    model: nn.Sequential
    max_epochs: int
    bs: int
    lr: float
    target_acc: int
    model_path: str
    data_path: str
    ds_mean: List[float]
    ds_std: List[float]
    img_size_x: int
    img_size_y: int
    log_step: int
    verbose: List[bool]
    dev_num: Optional[int]
    num_workers: int
    resume: bool
    resume_model_name: str
    test_data_path: str
    test_model_name: str
    prediction_model_name: str


def config_path_process(path: str) -> str:
    path = path.lstrip("./\\")
    path = path.rstrip("py")
    path = path.rstrip(".")
    path = path.replace("/", ".")
    path = path.replace("\\", ".")
    return path
