import torch
import torch.nn as nn
from utils import (
    get_train_transform,
    get_valid_transform,
    get_loaders,
    train_fn,
    compute_print_val_metrics,
)
from model.vit import VisionTransformer
from torch.utils.tensorboard import SummaryWriter
from constant import DEVICE
from config import XSMALLVITCONFIG


def main():
    cfg = XSMALLVITCONFIG
    cfg.print_config()

    writer = SummaryWriter()

    train_transform = get_train_transform(cfg.image_size[0], cfg.image_size[1])
    valid_transform = get_valid_transform(cfg.image_size[0], cfg.image_size[1])

    model = VisionTransformer(cfg).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.adam_beta_1, cfg.adam_beta_2),
    )

    trn_loader, val_loader = get_loaders(
        train_transform, valid_transform, cfg.batch_size
    )

    for epoch in range(cfg.num_epoch):
        train_loss = train_fn(trn_loader, model, optimizer, loss_fn, device=DEVICE)

        valid_loss, valid_accuracy = compute_print_val_metrics(
            val_loader, model, loss_fn=loss_fn, device=DEVICE
        )

        writer.add_scalars(
            "Loss", {"train_loss": train_loss, "valid_loss": valid_loss}, epoch
        )
        writer.add_scalars("Accuracy", {"valid_accuracy": valid_accuracy}, epoch)

    torch.save(model, "cat_dog_classifier.pt")


if __name__ == "__main__":
    main()
