from glob import glob
from pathlib import Path
import random
import os
import shutil


if __name__ == "__main__":
    row_data_path = Path(Path().absolute(), "data", "row_data")
    cat_items = glob(f"{row_data_path}/cat*.jpg")
    dog_items = glob(f"{row_data_path}/dog*.jpg")
    random.seed(1)

    if os.path.exists("data/train"):
        shutil.rmtree("data/train")
    os.mkdir("data/train")

    if os.path.exists("data/valid"):
        shutil.rmtree("data/valid")
    os.mkdir("data/valid")

    random.shuffle(cat_items)
    random.shuffle(dog_items)

    for i in range(int(12500 * 0.05)):  # 5% for the valid set
        cat_filename = cat_items[i].split("/")[-1]
        dog_filename = dog_items[i].split("/")[-1]

        shutil.copy(cat_items[i], f"data/valid/{cat_filename}")
        shutil.copy(dog_items[i], f"data/valid/{dog_filename}")

    for i in range(int(12500 * 0.05) + 1, 12500):
        cat_filename = cat_items[i].split("/")[-1]
        dog_filename = dog_items[i].split("/")[-1]

        shutil.copy(cat_items[i], f"data/train/{cat_filename}")
        shutil.copy(dog_items[i], f"data/train/{dog_filename}")
