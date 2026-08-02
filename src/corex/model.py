class CorexModel:

    def __init__(
        self,
        config
    ):

        self.config = config

        self.name = (
            config["name"]
        )


    def generate(
        self,
        prompt: str
    ):

        return (
            f"{self.name}: "
            f"{prompt}"
        )
