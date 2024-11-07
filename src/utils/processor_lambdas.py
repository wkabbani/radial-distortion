import re
import logging
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torchvision import transforms
from torchvision.transforms.functional import crop

from config.processor import Distort, FaceCrop, FixedL, Processor, RandomL
from processing.distort_model import OpticalProjection
from processing.face_dlib import dlib_face_detector
from processing.face_localization import face_crop
from processing.face_mediapipe import mediapipe_face_detector
from processing.utils import pil_to_mediapipe
from utils.distort_to_fn import distort_dm, distort_kb

__camel_to_snake__ = re.compile(r"(?<!^)(?=[A-Z])")

# Retrieve hydra logger
log = logging.getLogger(__name__)


def l_gen_from_distort(distort: Distort):
    if isinstance(distort, FixedL):
        return lambda: distort.l_value
    elif isinstance(distort, RandomL):
        """Generates random lambda values between min and max for nb_sample elements"""
        torch.manual_seed(seed=distort.seed if distort.seed else torch.seed())
        distrib = torch.distributions.uniform.Uniform(distort.min, distort.max)

        return lambda: round(distrib.sample().item(), 2)
    else:
        raise ValueError(
            f"wrong config type, should be {[__camel_to_snake__.sub('_', cls.__name__).lower() for cls in Distort.__subclasses__()]}"
        )


def distort_from_transform(transform: Distort.Transform):
    try:
        transform = Distort.Transform[transform]
        if transform == Distort.Transform.division:
            return distort_dm
        else:
            return distort_kb(op=OpticalProjection[transform.name])
    except KeyError:
        raise ValueError(
            f"wrong [Transform] type, should be {Distort.Transform._member_names_}"
        )


def detector_from_face_crop(face_crop: FaceCrop):
    members = FaceCrop.Detector.__members__
    if face_crop.detector not in members:
        raise ValueError(f"wrong [Detector] type, should be {members}")

    detector = FaceCrop.Detector[face_crop.detector]
    if detector == FaceCrop.Detector.dlib:
        return dlib_face_detector()
    if detector == FaceCrop.Detector.mediapipe:
        return mediapipe_face_detector(face_crop.detector_path)

    raise ValueError(f"Error retrieving detector from face_crop")


def face_crop_lambda(
    detector: Callable[..., Tuple[int, int, int, int]],
    expand: Callable[[Tuple[int, int, int, int]], Tuple[int, int, int, int]],
) -> transforms.Lambda:
    face_localizer = face_crop(detector=detector, expand=expand)

    def cropper(t: torch.Tensor):
        size = t.shape
        H, W = size[1], size[2]
        crop_box = face_localizer(pil_to_mediapipe(
            transforms.ToPILImage()(t)), W, H)
        x, y = crop_box[0], crop_box[1]
        w, h = crop_box[2] - x, crop_box[3] - y
        return crop(t, y, x, h, w)

    return transforms.Lambda(cropper)


def distort_lambda(
    l_gen: Callable[[], float],
    distorter: Callable[[torch.Tensor, int, int, float], torch.Tensor]
):
    """
    Distort the elements in the given resources directory given a distortion
    function and a lambda generator.

    Parameters
    ----------
    distort_fn: (torch.Tensor, int, int, float) -> torch.Tensor
        Function used to distort image.
        Takes as parameter: source, height, width and lambda value.
    l_gen : (int) -> float
        Function to get a lambda value.

        Takes an int n as a parameter, n is used to retrieve a lambda value for
        the n item. n is useful in this context because each element is processed
        asynchronously.
    """

    def trsfrms(t: torch.Tensor) -> torch.Tensor:
        imgs = torch.unsqueeze(t, dim=0)
        size = t.shape
        H, W = size[1], size[2]
        return distorter(imgs, H, W, l_gen())[0]

    return transforms.Lambda(trsfrms)


def processor_to_lambda(processor: Processor) -> Optional[transforms.Lambda]:
    if isinstance(processor, FaceCrop):
        detector = detector_from_face_crop(processor)

        def expand(a): return a
        if processor.expand is not None:
            expand = processor.expand.expander
            log.info(f'FaceCrop with expander {type(processor.expand)}')

        return face_crop_lambda(detector=detector, expand=expand)

    if isinstance(processor, Distort):
        l_gen = l_gen_from_distort(processor)
        distorter = distort_from_transform(processor.transform)

        log.info(f'Distort {type(processor)}')
        return distort_lambda(l_gen=l_gen, distorter=distorter)

    return None


def processors_to_lambdas(processors: Optional[Dict[int, Processor]]):
    lambdas: List[transforms.Lambda] = []

    if not processors:
        return lambdas

    for key in sorted(processors):
        result = processor_to_lambda(processor=processors[key])
        if result:
            lambdas.append(result)

    return lambdas
