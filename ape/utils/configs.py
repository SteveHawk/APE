from torch import nn
from typing import List, Optional


class Configs:
    num_classes: int
    gray_scale: bool
    model: nn.Sequential
    num_epochs: int
    bs: int
    lr: float
    target_acc: int
    model_path: str
    data_path: str
    ds_mean: List[float]
    ds_std: List[float]
    img_size_x: int
    img_size_y: int
    resume: bool
    resume_model_name: str
    verbose: List[bool]
    dev_num: Optional[int]
    num_workers: int
    test_data_path: str
    test_model_name: str
