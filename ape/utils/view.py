from torch import nn
from torch.tensor import Tensor

class View(nn.Module):
    def __init__(self):
        super(View, self).__init__()

    def forward(self, x: Tensor) -> Tensor:  # type: ignore
        return x.view(x.size(0), -1)
