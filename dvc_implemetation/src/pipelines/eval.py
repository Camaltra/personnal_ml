import torch
import json
from tqdm import tqdm
from pathlib import Path

from src.pipelines.pipeline import Pipeline
from src.pipelines.dataset import get_test_loader, get_valid_transform


class Evaluator(Pipeline):
    def __init__(
        self,
        model_save_fpath: Path,
        test_dir_path_img: Path,
        test_dir_path_mask: Path,
        patch_size: int,
        metric_fpath: Path,
    ) -> None:
        super().__init__()
        self.model_save_fpath = model_save_fpath
        self.test_dir_path_img = test_dir_path_img
        self.test_dir_path_mask = test_dir_path_mask
        self.metric_fpath = metric_fpath

        trms = get_valid_transform(patch_size)

        self.device = self._get_device()
        self.model = torch.load(self.model_save_fpath).to(self.device)

        self.loader = get_test_loader(trms)

    def _compute_test_metrics(self) -> tuple[int, int]:
        num_correct = 0
        num_pixels = 0
        dice_score = 0
        total_loss = 0

        loop = tqdm(self.loader)

        self.model.eval()
        with torch.no_grad():
            for _, (x, y) in enumerate(loop):
                x = x.to(self.device)
                y = y.to(self.device).unsqueeze(1)
                preds = self.model(x)
                activated_preds = torch.sigmoid(preds)
                activated_preds = (activated_preds > 0.5).float()
                num_correct += (activated_preds == y).sum()
                num_pixels += torch.numel(activated_preds)
                dice_score += (2 * (activated_preds * y).sum()) / (
                    (activated_preds + y).sum() + 1e-8
                )
        global_accuracy = num_correct / num_pixels * 100
        global_dice_score = dice_score / len(self.loader)
        total_loss /= len(self.loader)

        return global_accuracy.item(), global_dice_score.item()

    def run(self) -> None:
        acc, dice = self._compute_test_metrics()

        json.dump(
            obj={"acc": acc, "dice": dice}, fp=open(self.metric_fpath, "w+"), indent=4
        )
