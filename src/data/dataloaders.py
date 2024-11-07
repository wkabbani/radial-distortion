from typing import Callable, List, Optional, Tuple

from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms


def datatransform(
    resize: Tuple[int, int],
    mean: List[float],
    std: List[float],
    pre_transform: List[transforms.Lambda] = [],
) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.ToTensor(),
            *pre_transform,
            transforms.Resize(resize, antialias=True),  # type: ignore
            transforms.Normalize(
                mean=mean,
                std=std,
            ),
        ]
    )


def dataset(dir: str, transform: transforms.Compose) -> datasets.ImageFolder:
    return datasets.ImageFolder(dir, transform=transform)


def dataloader(
    batch_size: int = 16,
    num_workers: int = 12,
    shuffle: bool = True,
    collate_fn: Optional[Callable] = None,
):
    def dataloader(dataset: Dataset) -> DataLoader:
        return DataLoader(
            dataset=dataset,
            shuffle=shuffle,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_fn
        )

    return dataloader


if __name__ == "__main__":
    pass
