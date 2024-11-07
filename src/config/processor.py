from abc import abstractmethod
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple

from omegaconf import MISSING


@dataclass
class Processor:
    pass


@dataclass
class Expand:
    @abstractmethod
    def expander(self, p: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return p


@dataclass
class Fixed(Expand):
    expansion: int = 0

    def expander(self, p: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return (
            p[0] - self.expansion,
            p[1] - self.expansion,
            p[2] + self.expansion,
            p[3] + self.expansion,
        )


@dataclass
class FixedLTRB(Expand):
    left: int = 0
    top: int = 0
    right: int = 0
    bottom: int = 0

    def expander(self, p: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        return (
            p[0] - self.left,
            p[1] - self.top,
            p[2] + self.right,
            p[3] + self.bottom,
        )


@dataclass
class Ratio(Expand):
    ratio_w: float = 0
    ratio_h: float = 0

    def expander(self, p: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
        x_diff, y_diff = p[2] - p[0], p[3] - p[1]
        return (
            p[0] - int(x_diff * self.ratio_w),
            p[1] - int(y_diff * self.ratio_h),
            p[2] + int(x_diff * self.ratio_w),
            p[3],
        )


@dataclass
class FaceCrop(Processor):
    class Detector(str, Enum):
        mediapipe = auto()
        dlib = auto()

        def __str__(self) -> str:
            return self.name

    detector: Detector = Detector.mediapipe
    detector_path: str = MISSING
    expand: Optional[Expand] = None


@dataclass
class Distort(Processor):
    class Transform(str, Enum):
        division = auto()
        perspective = auto()
        stereographic = auto()
        equisolid = auto()
        orthogonal = auto()

        def __str__(self) -> str:
            return self.name

    transform: Transform = MISSING


@dataclass
class FixedL(Distort):
    l_value: float = 0.6


@dataclass
class RandomL(Distort):
    min: float = MISSING
    max: float = MISSING
    seed: Optional[int] = None
