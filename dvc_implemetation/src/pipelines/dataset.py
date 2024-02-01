import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from glob import glob
from torchvision import transforms
import torch


def get_train_transform(
    image_size: int,
    max_rotate: float,
    p_rotate: float,
    p_flip_vert: float,
    p_flip_honz: float,
) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=max_rotate, p=p_rotate),
            A.HorizontalFlip(p=p_flip_vert),
            A.VerticalFlip(p=p_flip_honz),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )


def get_valid_transform(image_size: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Normalize(
                mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], max_pixel_value=255.0
            ),
            ToTensorV2(),
        ]
    )


class RoadDataset(Dataset):
    def __init__(self, dataset_type: str, transform: None | A.Compose = None) -> None:
        self.items = [
            img_path.split("/")[-1]
            for img_path in glob(f"src/data/{dataset_type}/img/*.tif")
        ]
        self.path = f"src/data/{dataset_type}"
        self.transform = transform

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, ix: int) -> tuple[np.ndarray, np.ndarray]:
        img = np.array(Image.open(f"{self.path}/img/{self.items[ix]}").convert("RGB"))
        mask = np.array(Image.open(f"{self.path}/mask/{self.items[ix]}").convert("L"))

        if transforms is not None:
            augmentations = self.transform(image=img, mask=mask)
            img = augmentations["image"]
            mask = augmentations["mask"]

        return img, mask


def get_loaders(
    train_transform: A.Compose,
    valid_transform: A.Compose,
    random_state: int,
    batch_size: int
) -> tuple[DataLoader, DataLoader]:
    trn_ds = RoadDataset("train", train_transform)
    val_ds = RoadDataset("valid", valid_transform)
    generator = torch.Generator()
    generator.manual_seed(random_state)
    return DataLoader(
        trn_ds, batch_size=batch_size, shuffle=True, generator=generator
    ), DataLoader(val_ds, batch_size=batch_size, shuffle=False)


def get_test_loader(transformer: A.Compose) -> DataLoader:
    test_ds = RoadDataset("test", transformer)
    return DataLoader(test_ds, batch_size=16, shuffle=False)
