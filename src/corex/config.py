import json
from pathlib import Path


def load_config(
    path="configs/model.json"
):

    config_path = Path(path)

    with open(
        config_path,
        "r"
    ) as file:

        return json.load(file)
