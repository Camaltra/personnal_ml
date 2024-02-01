from pathlib import Path
from src.utils.decorator import parser

from box import ConfigBox

from src.pipelines.split import Splitter
from src.utils.folder_management import create_folders


def split(params: ConfigBox) -> None:
    data_dir = Path(params.transform.raw_dataset_dir)
    valid_dir_path_img = data_dir / params.split.valid_dir_path / "img"
    valid_dir_path_mask = data_dir / params.split.valid_dir_path / "mask"
    train_dir_path_img = data_dir / params.transform.train_dir_path / "img"
    train_dir_path_mask = data_dir / params.transform.train_dir_path / "mask"
    random = params.base.random_state

    create_folders([valid_dir_path_img, valid_dir_path_mask])

    prc_valid = params.split.prc_valid

    splitter = Splitter(
        valid_dir_path_img,
        valid_dir_path_mask,
        train_dir_path_img,
        train_dir_path_mask,
        prc_valid,
        random,
    )
    splitter.run()



@parser(prog_name="Split Pipeline", dscr="Split the data base on random var for train and valid")
def main(params) -> None:
    split(params)


if __name__ == "__main__":
    main()
    