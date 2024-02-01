import cv2
from PIL import Image
import numpy as np
import os
import logging
from pathlib import Path

from src.pipelines.pipeline import Pipeline


logging.basicConfig(level=logging.INFO)


class BuildDataset(Pipeline):
    def __init__(
        self,
        raw_data_path: Path,
        train_img_dir_path: Path,
        train_mask_dir_path: Path,
        test_img_dir_path: Path,
        test_mask_dir_path: Path,
        valid_idx_set: set[int],
        patch_size: int,
    ):
        super().__init__()
        self.raw_data_path = raw_data_path
        self.train_img_dir_path = train_img_dir_path
        self.train_mask_dir_path = train_mask_dir_path
        self.test_img_dir_path = test_img_dir_path
        self.test_mask_dir_path = test_mask_dir_path
        self.valid_idx_set = valid_idx_set
        self.patch_size = patch_size

        self.train_patch_count = 0
        self.test_patch_count = 0
        self.logger = logging.getLogger("BuildDataset")

    def _load_img_mask_by_idx(self, img_idx: int) -> tuple[np.array, np.array]:
        img = cv2.imread(f"{self.raw_data_path}/{img_idx}/Ottawa-{img_idx}.tif", 1)

        mask = np.array(
            Image.open(f"{self.raw_data_path}/{img_idx}/segmentation.png").convert("L")
        )
        mask = (mask != 255).astype(np.float32)[:, :, np.newaxis]

        return img, mask

    def _is_train_img(self, img_idx: int) -> bool:
        return not img_idx in self.valid_idx_set

    @staticmethod
    def _save_patch(output_path: str, patch: np.array):
        cv2.imwrite(output_path, patch)

    def _process_img(self, img_idx: int) -> None:
        is_train_img = self._is_train_img(int(img_idx))
        if is_train_img:
            output_img_path = self.train_img_dir_path
            output_mask_path = self.train_mask_dir_path
        else:
            output_img_path = self.test_img_dir_path
            output_mask_path = self.test_mask_dir_path

        img, mask = self._load_img_mask_by_idx(img_idx)

        height, width, _ = img.shape

        patch_idx = 0
        for y in range(0, height - self.patch_size + 1, self.patch_size):
            for x in range(0, width - self.patch_size + 1, self.patch_size):
                try:
                    patched_image = img[
                        y : y + self.patch_size, x : x + self.patch_size
                    ]
                    patched_mask = mask[
                        y : y + self.patch_size, x : x + self.patch_size
                    ]

                    output_patch_img_path = os.path.join(
                        f"{output_img_path}", f"{img_idx}_patch_{patch_idx}.tif"
                    )
                    output_patch_mask_path = os.path.join(
                        f"{output_mask_path}", f"{img_idx}_patch_{patch_idx}.tif"
                    )

                    self._save_patch(output_patch_img_path, patched_image)
                    self._save_patch(output_patch_mask_path, patched_mask)

                    if is_train_img:
                        self.train_patch_count += 1
                    else:
                        self.test_patch_count += 1
                    patch_idx += 1

                except Exception as e:
                    # TODO: Classify the exception
                    self.logger.error(
                        f"Error while trying to patch img <{img_idx}> at patch idx <{patch_idx}>, continue..."
                    )
                    self.logger.debug(e)

    def run(self) -> None:
        for folder_name in os.listdir(self.raw_data_path):
            self._process_img(folder_name)

        self.logger.info(
            f"Pipeline Succeded | num of train patches: <{self.train_patch_count}> | num of test patches: <{self.test_patch_count}>"
        )
