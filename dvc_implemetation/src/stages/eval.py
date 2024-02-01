from pathlib import Path
from box import ConfigBox

from src.pipelines.eval import Evaluator
from src.utils.decorator import parser


def evaluation(params: ConfigBox) -> None:
    data_dir = Path(params.transform.raw_dataset_dir)
    model_save_fpath = params.train.model_pickle_fpath
    test_dir_path_img = data_dir / params.transform.test_dir_path / "img"
    test_dir_path_mask = data_dir / params.transform.test_dir_path / "mask"

    metrics_fpath = Path(params.eval.metrics_file)
    metrics_fpath.parent.mkdir(parents=True, exist_ok=True)

    patch_size = params.transform.patch_size

    evaluator = Evaluator(
        model_save_fpath,
        test_dir_path_img,
        test_dir_path_mask,
        patch_size,
        metrics_fpath,
    )
    evaluator.run()


@parser(prog_name="Eval Pipeline", dscr="Eval the new trained Model")
def main(params) -> None:
    evaluation(params)
    

if __name__ == "__main__":
    main()