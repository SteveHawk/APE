import torch
from torch import nn
from torchvision import transforms
from PIL import Image
import os
import sys


class nn_train(object):
    class View(nn.Module):
        def __init__(self):
            super(View, self).__init__()

        def forward(self, x):
            return x.view(x.size(0), -1)


def preprocess(img, dev, transform_img_size_x, transform_img_size_y):
    img = img.convert("L")
    size = img.size
    if size[0] > size[1]:
        img = img.rotate(90)
    img = img.resize((transform_img_size_y, transform_img_size_x), Image.ANTIALIAS)

    transform_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [1], std = [0.5])
        ]
    )
    tensor = transform_tensor(img)
    return tensor.view(-1, 1, transform_img_size_x, transform_img_size_y).to(dev)


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


def predict(images):
    # Some magic stuff
    sys.modules["nn_train"] = nn_train

    # Params
    model_path = "./models/"
    model_name = "target.pth.tar"
    transform_img_size_x = 200
    transform_img_size_y = 200
    dev_num = None

    # Device settings
    if params.dev_num is None:
        dev = torch.device("cpu")
    elif torch.cuda.is_available():
        dev = torch.device(f"cuda:{dev_num}")
        torch.backends.cudnn.benchmark = True
    else:
        dev = torch.device("cpu")
    print(f"Device: {dev}")

    # Model loading
    model = load_cp(model_path, model_name, dev).to(dev)
    model.eval()
    print(model)
    print("Model Loading Done.")

    labels = list()
    with torch.no_grad():
        for img in images:
            outputs = model(preprocess(img, dev, transform_img_size_x, transform_img_size_y))
            _, predicted = torch.max(outputs.data, 1)
            labels.append(predicted.item())
    
    return labels


def example():
    imgs = list()
    for i in range(1, 11):
        imgs.append(Image.open(f"./ds/img ({i}).png"))

    result = predict(imgs)
    
    for i in range(1, 11):
        print(f"img ({i}).png  {result[i - 1]}")


if __name__ == "__main__":
    example()
