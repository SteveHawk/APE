from torch import nn


class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x: object) -> object:
        return x.view(x.size(0), -1)
