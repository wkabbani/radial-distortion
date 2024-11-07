from typing import Callable

import torch
from torch.nn.functional import grid_sample

from processing.distort_model import (
    OpticalProjection,
    division_model,
    division_model_crop,
    generic_kannala_brandt,
    naive_crop,
)


def distort_kb(
    op: OpticalProjection
) -> Callable[[torch.Tensor, int, int, float], torch.Tensor]:
    gkb = generic_kannala_brandt(op)

    def compute(imgs: torch.Tensor, H: int, W: int, l: float) -> torch.Tensor:
        fisheye_grid = gkb(H, W, torch.tensor([0, 0]), [l])
        distorted_model = grid_sample(imgs, fisheye_grid, align_corners=True)
        return naive_crop(fisheye_grid, distorted_model, W, H)

    return compute


def distort_dm(imgs: torch.Tensor, H: int, W: int, l: float) -> torch.Tensor:
    fisheye_grid = division_model(H, W, torch.tensor([0, 0]), lambdas=[l])
    distorted_model = grid_sample(imgs, fisheye_grid, align_corners=True)
    return division_model_crop(distorted_model, W, H, l=l)
