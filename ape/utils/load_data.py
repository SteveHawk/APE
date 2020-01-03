import torch
from torch import Tensor
from torchvision import transforms  # type: ignore
from torchvision.datasets import ImageFolder  # type: ignore
from torch.utils.data import DataLoader, random_split

from PIL import Image  # type: ignore
from typing import Tuple, List, Callable, Iterator

from .configs import Configs


class WrappedDataLoader:
    def __init__(self, dl: DataLoader, func: Callable[[Tensor, Tensor],
                                        Tuple[Tensor, Tensor]]) -> None:
        self.dl = dl
        self.func = func

    def __len__(self) -> int:
        return len(self.dl)

    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        batches = iter(self.dl)
        for b in batches:
            yield (self.func(*b))


def transform(img_size_x: int, img_size_y: int, ds_mean: List[float], ds_std: List[float],
                                        gray_scale: bool) -> Callable[[Image.Image], Tensor]:
    def _transform(img: Image.Image) -> Tensor:
        if gray_scale:
            img = img.convert("L")
        size = img.size
        img = img.resize((img_size_x, img_size_y), Image.ANTIALIAS)

        transform_tensor = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=ds_mean, std=ds_std)
            ]
        )
        return transform_tensor(img)
    return _transform


def preprocess(img_size_x: int, img_size_y: int, dev: torch.device, gray_scale: bool) \
                                -> Callable[[Tensor, Tensor], Tuple[Tensor, Tensor]]:
    def _preprocess(x: Tensor, y: Tensor) -> Tuple[Tensor, Tensor]:
        return x.view(-1, 1 if gray_scale else 3, img_size_x, img_size_y).to(dev), y.to(dev)
    return _preprocess


def load_data(Configs: Configs, dev: torch.device) -> Tuple[WrappedDataLoader, WrappedDataLoader]:
    full_ds = ImageFolder(Configs.data_path, transform=transform(Configs.img_size_x, Configs.img_size_y,
        Configs.ds_mean, Configs.ds_std, Configs.gray_scale))
    print("ImageFolder done.")
    print(full_ds.class_to_idx)

    train_size = int(0.8 * len(full_ds))
    valid_size = len(full_ds) - train_size
    train_ds, valid_ds = random_split(full_ds, [train_size, valid_size])
    print("Dataset 8/2 random split done.")

    _train_dl = DataLoader(train_ds, batch_size=Configs.bs, shuffle=True, num_workers=Configs.num_workers)
    _valid_dl = DataLoader(valid_ds, batch_size=Configs.bs, num_workers=Configs.num_workers)
    print("DataLoader done.")

    train_dl = WrappedDataLoader(_train_dl, preprocess(Configs.img_size_x, Configs.img_size_y,
        dev, Configs.gray_scale))
    valid_dl = WrappedDataLoader(_valid_dl, preprocess(Configs.img_size_x, Configs.img_size_y,
        dev, Configs.gray_scale))
    print("WrappedDataLoader done.")

    return train_dl, valid_dl
