import torch

from corex.config import load_config
from corex.model import CorexModel


def test_model():

    config = load_config()

    model = CorexModel(config)

    tokens = torch.randint(
        0,
        config["model"]["vocab_size"],
        (1, 32)
    )

    output = model(tokens)

    assert output.shape == (
        1,
        32,
        config["model"]["vocab_size"]
    )
