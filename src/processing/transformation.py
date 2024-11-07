import os
from pathlib import Path
from typing import Any, Callable, Optional

import torch
import torchvision.transforms as transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.folder import pil_loader
from torchvision.utils import save_image
from tqdm import tqdm

from data.dataloaders import dataloader

SUPPORTED_IMG_EXTENSIONS = [".jpeg", ".png", ".jpg"]


class CustomVisionDataset(VisionDataset):
    def __init__(
        self,
        root: str,
        transform: Optional[Callable[..., Any]] = None,
    ) -> None:
        super().__init__(root, None, transform, None)
        directory = Path(os.path.expanduser(self.root))
        self.samples = sorted(
            item.resolve()
            for item in directory.rglob("*")
            if item.is_file() and item.suffix.upper() in [ext.upper() for ext in SUPPORTED_IMG_EXTENSIONS]
        )
        self.loader = pil_loader

    def __getitem__(self, index):
        sample = self.loader(self.samples[index])
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.samples)


def save_tensor(img: torch.Tensor, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_image(img, str(path))


def apply_transformation(
    source_dir: str,
    dest_dir: str,
    transform: transforms.Compose,
):
    # log = logging.getLogger(__name__)

    dataset = CustomVisionDataset(root=source_dir, transform=transform)
    loader = dataloader(shuffle=False, collate_fn=lambda a: a)(dataset=dataset)

    for i, batch in tqdm(enumerate(loader), total=len(loader)):
        for idx, transformed in enumerate(batch):

            new_path = Path(dest_dir).joinpath(dataset.samples[(
                i * 16) + idx].relative_to(source_dir))

            save_tensor(
                img=transformed,
                path=new_path
            )


if __name__ == "__main__":
    pass
