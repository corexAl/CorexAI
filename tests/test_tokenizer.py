from corex.tokenizer import Tokenizer


def test_tokenizer():

    tokenizer = Tokenizer()

    text = "hello COREX"

    encoded = tokenizer.encode(
        text
    )

    decoded = tokenizer.decode(
        encoded
    )

    assert decoded == "hello corex"
