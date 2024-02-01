from pathlib import Path
import os


def create_folders(folder_paths: list[Path]) -> None:
    for folder_path in folder_paths:
        if not folder_path.exists():
            os.makedirs(folder_path)
