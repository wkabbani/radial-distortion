from typing import Tuple

import mediapipe as mp
import numpy as np
import torch
from PIL import Image


def rect_to_bb(rect) -> Tuple[int, int, int, int]:
    """Take a bounding predicted by dlib and convert it
    to the format (x, y, w, h) as we would normally do
    with OpenCV"""
    x = rect.left()
    y = rect.top()
    w = rect.right() - x
    h = rect.bottom() - y
    # return a tuple of (x, y, w, h)
    return (x, y, w, h)


def shape_to_np(shape, dtype="int"):
    # initialize the list of (x, y)-coordinates
    coords = np.zeros((68, 2), dtype=dtype)
    # loop over the 68 facial landmarks and convert them
    # to a 2-tuple of (x, y)-coordinates
    for i in range(0, 68):
        coords[i] = (shape.part(i).x, shape.part(i).y)
    # return the list of (x, y)-coordinates
    return coords


def pil_to_mediapipe(img: Image.Image):
    return numpy_to_mediapipe(np.asarray(img))


def tensor_to_mediapipe(t: torch.Tensor):
    """TODO : Check why this is not working"""
    return numpy_to_mediapipe(
        np.moveaxis((t * 255).to(dtype=torch.uint8).numpy(), 0, -1)
    )


def numpy_to_mediapipe(n: np.ndarray):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=n)
