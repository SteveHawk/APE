from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
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


def load_data(data_path, bs, transform_img_size_x, transform_img_size_y, dev, num_workers, ds_mean, ds_std) -> Tuple[WrappedDataLoader]:
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
