from spamdetector.lexicator import *


def test_tokenize_content():
    x = tokenize_content("a-.';]]'", "@#@#b;.,")
    assert x == ['a', 'b']


def test_lemmatize_content():
    x = lemmatize_content("Extraordinary painters paint craziest fantasies",
                          "Three little marines")
    assert x == "Extraordinary painter paint craziest fantasy Three little " \
                "marine"


def test_remove_stop_words():
    x = remove_stopwords("mary is dedicated to medal of honour")
    assert x == "mary dedicated medal honour"


if __name__ == '__main__':
    test_tokenize_content()
    test_lemmatize_content()
    test_remove_stop_words()
