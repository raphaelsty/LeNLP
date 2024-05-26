from rslenlp import (
    rsvectorize_char_ngrams_many,
    rsvectorize_char_wb_ngrams_many,
    rsvectorize_split_words_many,
)

__all__ = ["count"]


def count(
    x: str | list[str],
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 1),
    normalize: bool = True,
    stop_words: list[str] = None,
    sort: bool = False,
) -> dict[str, int]:
    """Count the frequency of words in a text or in a list of texts. Tokens are unordered within
    the same text.

    Parameters
    ----------
    x
        str or list of str.
    analyzer
        {word, char, char_wb}, default=word.
        Whether the feature should be made of word n-gram or character n-grams. Option
        char_wb creates character n-grams only from text inside word boundaries;
        n-grams at the edges of words are padded with space.
    ngram_range
        tuple (min_n, max_n), default=1.
        The lower and upper boundary of the range of n-values for different n-grams to
        be extracted. All values of n such that min_n <= n <= max_n will be used.
    normalize
        bool, default=True.
        Whether to normalize the text before counting. It will lowercase the text and remove
        punctuation.
    stop_words
        list of str, default=None.
        A list of stop words that will be removed from the text.

    Examples
    --------
    >>> from lenlp import counter

    >>> counter.count("Hello, world!", sort=True)
    {'hello': 1, 'world': 1}

    >>> counter.count("Hello, world!", ngram_range=(2, 2), sort=True, normalize=False)
    {'Hello, world!': 1}

    >>> counter.count(["Hello, world!", "How are you?"], stop_words=["are", "you"], sort=True)
    [{'hello': 1, 'world': 1}, {'how': 1}]

    >>> counter.count(["Hello, world!", "hello"], analyzer="char_wb", ngram_range=(3, 7), stop_words=["hello"], sort=True)
    [{'orl': 1, 'orld': 1, 'rld': 1, 'wor': 1, 'worl': 1, 'world': 1}, {}]

    >>> counter.count("Hello, world!", analyzer="char_wb", ngram_range=(3, 7), sort=True)
    {' wo': 1, ' wor': 1, ' worl': 1, ' world': 1, 'ell': 1, 'ello': 1, 'ello ': 1, 'ello w': 1, 'ello wo': 1, 'hel': 1, 'hell': 1, 'hello': 1, 'hello ': 1, 'hello w': 1, 'llo': 1, 'llo ': 1, 'llo w': 1, 'llo wo': 1, 'llo wor': 1, 'lo ': 1, 'lo w': 1, 'lo wo': 1, 'lo wor': 1, 'lo worl': 1, 'o w': 1, 'o wo': 1, 'o wor': 1, 'o worl': 1, 'o world': 1, 'orl': 1, 'orld': 1, 'rld': 1, 'wor': 1, 'worl': 1, 'world': 1}

    >>> counter.count("Hello, world!", analyzer="char", ngram_range=(3, 7), sort=True)
    {' wo': 1, ' wor': 1, ' worl': 1, ' world': 1, 'ell': 1, 'ello': 1, 'ello ': 1, 'ello w': 1, 'ello wo': 1, 'hel': 1, 'hell': 1, 'hello': 1, 'hello ': 1, 'hello w': 1, 'llo': 1, 'llo ': 1, 'llo w': 1, 'llo wo': 1, 'llo wor': 1, 'lo ': 1, 'lo w': 1, 'lo wo': 1, 'lo wor': 1, 'lo worl': 1, 'o w': 1, 'o wo': 1, 'o wor': 1, 'o worl': 1, 'o world': 1, 'orl': 1, 'orld': 1, 'rld': 1, 'wor': 1, 'worl': 1, 'world': 1}

    >>> counter.count(["Hello, world!", "hello"], analyzer="char", ngram_range=(3, 7), stop_words=["hello"], sort=True)
    [{'orl': 1, 'orld': 1, 'rld': 1, 'wor': 1, 'worl': 1, 'world': 1}, {}]

    """
    return_string = True if isinstance(x, str) else False
    x = [x] if isinstance(x, str) else x
    n_sizes = list(range(ngram_range[0], ngram_range[1] + 1))

    match analyzer:
        case "word":
            y = rsvectorize_split_words_many(
                x, n_sizes=n_sizes, stop_words=stop_words, normalize=normalize
            )
        case "char":
            y = rsvectorize_char_ngrams_many(
                x, n_sizes=n_sizes, stop_words=stop_words, normalize=normalize
            )

        case "char_wb":
            y = rsvectorize_char_wb_ngrams_many(
                x, n_sizes=n_sizes, stop_words=stop_words, normalize=normalize
            )

    if sort:
        y = [dict(sorted(d.items())) for d in y]

    return y[0] if return_string else y
