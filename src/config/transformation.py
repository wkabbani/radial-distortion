from dataclasses import dataclass, field
from typing import Dict
from omegaconf import MISSING
from config.processor import Processor


@dataclass
class Transformation:
    source: str = MISSING
    destination: str = MISSING
    transforms: Dict[int, Processor] = MISSING
