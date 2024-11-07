from typing import Callable, Tuple
import numpy as np
import dlib
from PIL import Image


def dlib_face_detector(
    resize: int = 256,
    n_layers: int = 0,
) -> Callable[[Image.Image], Tuple[int, int, int, int]]:
    def detect(image: Image.Image) -> Tuple[int, int, int, int]:
        width, height = image.size
        min_dim = min(width, height)
        ratio = 1
        c_image = image
        if min_dim > resize:
            ratio = resize / min_dim
            c_image = image.resize(size=(round(width * ratio), round(height * ratio)))

        np_img = np.asarray(c_image.convert("L"))
        detector = dlib.get_frontal_face_detector()  # type: ignore
        rects = detector(np_img, n_layers)
        if len(rects) == 0:
            return 0, 0, width, height

        rect = rects[0]
        return (
            round(rect.left() / ratio),
            round(rect.top() / ratio),
            round(rect.right() / ratio),
            round(rect.bottom() / ratio),
        )

    return detect
