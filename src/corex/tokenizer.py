class Tokenizer:

    def __init__(self):

        self.vocab = {}


    def encode(
        self,
        text: str
    ):

        return text.split()


    def decode(
        self,
        tokens
    ):

        return " ".join(tokens)
