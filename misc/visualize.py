from typing import Union, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL.Image import Image


def to_HWC(img):
    if isinstance(img, np.ndarray) and img.shape[0] in [1, 3]:
        return img.transpose((1, 2, 0))
    elif isinstance(img, torch.Tensor) and img.shape[0] in [1, 3]:
        return img.permute((1, 2, 0)).detach().cpu().numpy()
    elif isinstance(img, torch.Tensor) and len(img.shape) == 2:
        return img.detach().cpu().numpy()
    elif isinstance(img, Image):
        return np.asarray(img)
    else:
        return img


def imshow(
    img: Union[torch.Tensor, np.ndarray], title: Optional[str] = None, cmap: Optional[str] = None, alpha: float = 1.0
):
    # assert len(img.shape) < 4, "Please provide a max of 3D input"
    img = to_HWC(img)
    plt.imshow(img, cmap=cmap, alpha=alpha)
    # plt.axis("off")
    if title is not None:
        plt.title(title)
