import torch
from config import ViTConfig


DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
