import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from PIL import Image
from glob import glob
from torchvision import transforms
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import cv2


class CatVsDogDataset(Dataset):
    def __init__(self, type: str, transform: None | A.Compose = None) -> None:
        self.path = Path(Path().absolute(), "..", "data", type)
        print(self.path)
        self.items = [
            img_path.split("/")[-1] for img_path in glob(f"{self.path}/*.jpg")
        ]
        self.transform: A.Compose | None = transform or A.Compose(
            [
                A.Resize(height=256, width=256, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, ix: int) -> tuple[torch.Tensor, torch.Tensor]:
        img = np.array(Image.open(f"{self.path}/{self.items[ix]}").convert("RGB"))
        target = 1 if self.items[ix].startswith("dog") else 0
        if transforms is not None:
            img = self.transform(image=img)["image"]
        return img.float(), torch.tensor([target]).float()

    def __len__(self) -> int:
        return len(self.items)
