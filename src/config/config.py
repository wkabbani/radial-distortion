from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple

from omegaconf import MISSING

from config.transformation import Transformation

@dataclass
class TransformsConfig:
    resize: Tuple[int, int] = MISSING
    mean: List[float] = MISSING
    std: List[float] = MISSING


class RunType(str, Enum):
    transformation = auto()


@dataclass
class Config:
    run: RunType = MISSING
    results: Optional[str] = MISSING
    transforms: TransformsConfig = field(default_factory=TransformsConfig)
    transformation: Transformation = field(default_factory=Transformation)
