#!/usr/bin/env python3

import os
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List

import torch
import hydra
from omegaconf import MISSING, OmegaConf
from torchvision import transforms

from config.config import Config, RunType
from processing.transformation import apply_transformation
from utils.processor_lambdas import processors_to_lambdas

# Retrieve hydra logger
__main_logger__ = logging.getLogger(__name__)

__camel_to_snake__ = re.compile(r"(?<!^)(?=[A-Z])")


def dict_to_str(
    pdict: Dict[str, Any], add_if_missing: bool = False, exclude: List[str] = []
) -> str:
    return ",".join(
        [
            f"[{key}:{dict_to_str(res, add_if_missing=add_if_missing, exclude=exclude)}]"
            if isinstance(res, dict)
            else f"{key}={res}"
            for key, res in pdict.items()
            if key not in exclude
            and (add_if_missing or (not res or res != f"{MISSING}"))
        ]
    )


def dump_resuls(results: Dict[str, Any], cfg: Config, f_name: str):
    if cfg.results is None:
        return

    p = Path(cfg.results)
    p.mkdir(parents=True, exist_ok=True)
    with open(p / f_name, "w") as fp:
        json.dump(results, fp)
        fp.flush()
    __main_logger__.info(f"Results saved at {p / f_name}")


def dump_config(cfg: Config):
    if cfg is None or cfg.results is None:
        return

    f_name = os.path.join(cfg.results, 'config.json')
    with open(f_name, "w") as fp:
        json.dump(cfg.__dict__, fp)
        fp.flush()
    __main_logger__.info(f"Results saved at {f_name}")


def __transformation_main__(cfg: Config):
    src = cfg.transformation.source
    dst = cfg.transformation.destination

    tfms = [
        transforms.Lambda(transforms.Resize(size=1000)),
        transforms.Lambda(transforms.ToTensor()),
        *processors_to_lambdas(cfg.transformation.transforms),
    ]

    __main_logger__.info(
        f"Transforming source [{src}] into [{dst}] with [{[type(tfm) for tfm in tfms]}]"
    )

    apply_transformation(
        src,
        dst,
        transforms.Compose(tfms)
    )


@hydra.main(config_path="conf", config_name="config", version_base=None)
def __app__(cfg: Config):
    cfg = hydra.utils.instantiate(cfg)

    # uncomment this if you don't get issues when using the gpu
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = 'cpu'

    mains: Dict[RunType, Callable[[Config], None]] = {
        RunType.transformation: __transformation_main__,
    }

    try:
        runner = mains[RunType[f"{cfg.run}"]]
    except KeyError:
        __main_logger__.exception(
            f"Wrong run type run={cfg.run}, should be {RunType._member_names_}"
        )
        return

    try:
        runner(cfg)
    except:
        __main_logger__.exception(f"caught exception running [{cfg.run}]")


if __name__ == "__main__":

    def split_resolver(n: int, split: str, origin: str, *, _parent_):
        return "-".join(origin.split(split)[-n:]).lower()

    OmegaConf.register_new_resolver("split", split_resolver)

    __app__()
