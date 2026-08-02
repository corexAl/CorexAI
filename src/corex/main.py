from .config import load_config
from .model import CorexModel


def main():

    config = load_config()

    model = CorexModel(
        config
    )

    print(
        f"{config['name']} {config['version']} loaded"
    )


if __name__ == "__main__":
    main()
