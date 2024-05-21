from rsantelope import rsnormalize, rsnormalize_many

__all__ = ["normalize", "normalize_many"]


def normalize(text: str) -> str:
    """Lowercase, remove punctation and unidecode single text.

    Examples
    --------
    >>> from antelope import normalizer

    >>> normalizer.normalize("Hello, world!")
    'hello world'

    """
    return rsnormalize(text)


def normalize_many(texts: list[str]) -> str:
    """Lowercase, remove punctation and unidecode list of texts.

    Examples
    --------
    >>> from antelope import normalizer

    >>> normalizer.normalize_many(["Hello, world!", "Hello, world!"])
    ['hello world', 'hello world']

    """
    return rsnormalize_many(texts)
