import json
from pathlib import Path


class Tokenizer:

    def __init__(
        self,
        vocab_path="src/corex/vocab.json"
    ):

        self.vocab_path = Path(
            vocab_path
        )

        self.vocab = {}

        self.inverse_vocab = {}

        self.load_vocab()



    def load_vocab(self):

        if self.vocab_path.exists():

            with open(
                self.vocab_path,
                "r",
                encoding="utf-8"
            ) as file:

                self.vocab = json.load(file)

        else:

            self.create_default_vocab()


        self.inverse_vocab = {
            value: key
            for key, value
            in self.vocab.items()
        }



    def create_default_vocab(self):

        tokens = [
            "<pad>",
            "<unk>",
            "<bos>",
            "<eos>"
        ]

        self.vocab = {
            token: index
            for index, token
            in enumerate(tokens)
        }

        self.save_vocab()



    def save_vocab(self):

        self.vocab_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with open(
            self.vocab_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                self.vocab,
                file,
                indent=4
            )



    def add_token(
        self,
        token: str
    ):

        if token not in self.vocab:

            self.vocab[token] = len(
                self.vocab
            )



    def encode(
        self,
        text: str
    ):

        tokens = text.lower().split()

        ids = []

        for token in tokens:

            if token not in self.vocab:

                self.add_token(token)

            ids.append(
                self.vocab[token]
            )

        return ids



    def decode(
        self,
        ids
    ):

        tokens = []

        for index in ids:

            token = self.inverse_vocab.get(
                index,
                "<unk>"
            )

            tokens.append(token)


        return " ".join(tokens)
