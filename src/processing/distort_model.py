from enum import Enum
from fractions import Fraction
import torch

from math import sqrt
from torchvision.transforms import CenterCrop
from typing import Callable, List, Optional, Tuple
from functools import partial


def equationroots(a, b, c) -> Tuple[float, float]:
    """Solves a quadratic equation which has two solution"""
    dis = b * b - 4 * a * c
    sqrt_val = sqrt(abs(dis))

    return (-b + sqrt_val) / (2 * a), (-b - sqrt_val) / (2 * a)


def meshgrid(height, width) -> torch.Tensor:
    xx = torch.linspace(-1, 1, width)
    yy = torch.linspace(-1, 1, height)
    gridy, gridx = torch.meshgrid(yy, xx, indexing="ij")  # create identity grid
    grid = torch.stack([gridx, gridy], dim=-1)
    return grid


def naive_crop(grid: torch.Tensor, img: torch.Tensor, W: int, H: int) -> torch.Tensor:
    """Crop the image by searching the first pixel on the diagonal not mapped
    to the the given image from the center using dichotomy."""
    m = Fraction(H / W).limit_denominator(10)
    x, fin_x = int(W / 2), 0

    while True:
        if x <= fin_x:
            break

        mil = int((x + fin_x) / 2)
        fx = round(m * mil)
        if mil <= 0 or fx <= 0:
            x, fx = 0, 0
            break

        if abs(grid[0, fx, mil, 0]) > 1 or abs(grid[0, fx, mil, 1]) > 1:
            fin_x = mil + 1
        else:
            x = mil - 1

    return CenterCrop((H - round(m * x) * 2, W - x * 2))(img)


def division_model_crop(img: torch.Tensor, W: int, H: int, l: float) -> torch.Tensor:
    """Crop the image distorted with the division model.

    Parameters
    ----------
    img : Tensor
        Barrel distorted image Tensor.
    W : int
        Image width.
    H : int
        Image height
    l : float
        Distortion value, should be less than 0.

    Returns
    -------
    Tensor
        Cropped image.
    """

    assert l < 0

    ru = sqrt(2)
    root1, root2 = equationroots(1, -(1 / (l * ru)), 1 / l)
    root = max(root1, root2)

    coefd = root / ru
    transform = CenterCrop((round(coefd * H), round(coefd * W)))
    return transform(img)


def division_model(
    height: int,
    width: int,
    center: torch.Tensor,
    lambdas: List[float]
) -> torch.Tensor:
    """Computes the distorted grid given size and distortion parameters

    Parameters
    ----------
    height : int
        Height of the Tensor
    width : str
        Width of the Tensor
    center : Tensor
        Center of the distortion
    lambdas: List[float]
        List of lambda values. Their position in the list represent their
        position in the equation: lambda0 * r + lambda1 + r**2 + ...

    Returns
    -------
    Tensor
        Distorted grid
    """

    grid = meshgrid(height=height, width=width)
    center = center #.to(device=device)

    d = grid - center
    rd2 = (d**2).sum(dim=-1)

    d_sums = 0
    for i, l in enumerate(lambdas):
        d_sums += 0 if l == 0 else (rd2 ** (i + 1)).unsqueeze(-1) * l

    grid = center + (d / (1 + d_sums))
    return grid.unsqueeze(0)  # unsqueeze(0) since the grid needs to be 4D.


class OpticalProjection(partial, Enum):
    """Different type of Optical Projection used for the Kannala-Brandt model"""

    perspective = partial(lambda rd: torch.atan(rd))
    stereographic = partial(lambda rd: 2 * torch.atan(rd / 2))
    equisolid = partial(lambda rd: 2 * torch.asin(rd / 2))
    orthogonal = partial(lambda rd: torch.asin(rd))

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return self.value(r)


def generic_kannala_brandt(
    op: OpticalProjection,
) -> Callable[[int, int, torch.Tensor, List[float]], torch.Tensor]:
    """Kannala-Brandt distortion model

    Parameters
    ----------
    op : OpticalProjection
        Type of projection which will be used.

    Returns
    -------
    (height: int, width: int, center: torch.Tensor, lambdas: List[float]) -> Tensor
        The computation function for the given OpticalProjection.
    """

    def compute(
        height: int, width: int, center: torch.Tensor, lambdas: List[float]
    ) -> torch.Tensor:
        grid = meshgrid(height=height, width=width)

        d = grid - center #.to(device=device)
        r = (d**2).sum(dim=-1).sqrt()

        theta = op(r)
        thetad = theta
        for i, l in enumerate(lambdas):
            thetad += l * (theta ** (((i + 1) * 2) + 1))

        grid *= thetad.unsqueeze(-1) / r.unsqueeze(-1)

        return grid.unsqueeze(0)

    return compute


if __name__ == "__main__":
    pass
