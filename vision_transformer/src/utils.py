import torch
from albumentations.pytorch import ToTensorV2
import albumentations as A
from torch.utils.data import DataLoader
import cv2
from tqdm import tqdm
from data import CatVsDogDataset


def get_train_transform(image_height: int, image_width: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(
                height=image_height, width=image_width, interpolation=cv2.INTER_LINEAR
            ),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ]
    )


def get_valid_transform(image_height: int, image_width: int) -> A.Compose:
    return A.Compose(
        [
            A.Resize(
                height=image_height, width=image_width, interpolation=cv2.INTER_LINEAR
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            ),  # Value taken from the ViT in the Pytorch lib
            ToTensorV2(),
        ]
    )


def get_loaders(
    train_transform: A.Compose, valid_transform: A.Compose, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    trn_ds = CatVsDogDataset("train", train_transform)
    val_ds = CatVsDogDataset("valid", valid_transform)
    return DataLoader(
        trn_ds, batch_size=batch_size, shuffle=True, drop_last=True
    ), DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=True)


def compute_print_val_metrics(
    loader: DataLoader,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    device: str = "mps",
) -> tuple[float, float]:
    total_loss = 0

    model.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float()
            total_well_classified = (preds == y).sum()
            total_loss += loss_fn(preds, y.float()).item()
    model.train()
    accuracy = total_well_classified / len(loader)
    total_loss /= len(loader)
    print(f"ACCURACY {accuracy:.2f}%")

    return total_loss, accuracy


def train_fn(
    loader: DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    device: str,
) -> float:
    loop = tqdm(loader)
    total_loss = 0

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().to(device)

        predictions = model(data)
        loss = loss_fn(predictions, targets)
        total_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loop.set_postfix(loss=loss.item())

    return total_loss / len(loader)
