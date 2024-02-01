from argparse import ArgumentParser
from typing import Callable
import yaml
from box import ConfigBox


def load_params(params_file: str) -> ConfigBox:
    with open(params_file, "r") as f:
        params = yaml.safe_load(f)
        params = ConfigBox(params)
    return params


def parser(
    prog_name: str, dscr: str
) -> Callable[[Callable], Callable]:
    def decorator(function):
        def new_function(*args, **kwargs):            
            prs = ArgumentParser(
                prog=prog_name,
                description=dscr,
            )
            prs.add_argument("--config", dest="config", required=True)
            args = prs.parse_args()
            params_path = args.config
            params = load_params(params_path)
            function(params)

        return new_function

    return decorator