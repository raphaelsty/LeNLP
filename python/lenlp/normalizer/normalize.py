from lenlp._rslenlp import rsnormalize, rsnormalize_many

__all__ = ["normalize"]


def normalize(x: str | list[str]) -> str:
    """Lowercase, remove punctation and unidecode single text.

    Examples
    --------
    >>> from lenlp import normalizer

    >>> normalizer.normalize("Hello, world!")
    'hello world'

    >>> normalizer.normalize(["Hello, world!", "How are you?"])
    ['hello world', 'how are you']

    """
    return rsnormalize(x) if isinstance(x, str) else rsnormalize_many(x)
