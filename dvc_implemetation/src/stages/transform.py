from pathlib import Path

from box import ConfigBox

from src.pipelines.transform import BuildDataset

from src.utils.folder_management import create_folders
from src.utils.decorator import parser


def transform(params: ConfigBox) -> None:
    data_dir = Path(params.transform.raw_dataset_dir)
    raw_data_path = data_dir / params.transform.raw_dataset_fname

    train_dir_path_img = data_dir / params.transform.train_dir_path / "img"
    train_dir_path_mask = data_dir / params.transform.train_dir_path / "mask"
    test_dir_path_img = data_dir / params.transform.test_dir_path / "img"
    test_dir_path_mask = data_dir / params.transform.test_dir_path / "mask"

    create_folders(
        [train_dir_path_img, train_dir_path_mask, test_dir_path_img, test_dir_path_mask]
    )

    valid_idx_set = set(params.transform.test_idx)
    patch_size = params.transform.patch_size

    builder = BuildDataset(
        raw_data_path,
        train_dir_path_img,
        train_dir_path_mask,
        test_dir_path_img,
        test_dir_path_mask,
        valid_idx_set,
        patch_size,
    )
    builder.run()


@parser(prog_name="Transform Pipeline", dscr="Split Naturally the data | Patchify it")
def main(params) -> None:
    transform(params)
    

if __name__ == "__main__":
    main()
    