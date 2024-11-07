from typing import Any, Callable, Tuple

import cv2

from processing.utils import shape_to_np


def facial_landmarks(image, rect, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shape = predictor(gray, rect)
    return shape_to_np(shape)


def face_crop(
    detector: Callable[..., Tuple[int, int, int, int]],
    expand: Callable[[Tuple[int, int, int, int]], Tuple[int, int, int, int]],
):
    def localize(image, width: int, height: int):
        box = expand(detector(image))
        return (
            max(0, box[0]),
            max(0, box[1]),
            min(width, box[2]),
            min(height, box[3]),
        )

    return localize
