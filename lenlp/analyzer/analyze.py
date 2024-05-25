from rslenlp import rschar_ngrams_many, rschar_wb_ngrams_many, rssplit_words_many

__all__ = ["analyze"]


def analyze(
    x: str | list[str],
    analyzer: str = "word",
    ngram_range: tuple[int, int] = (1, 1),
) -> str | list[str]:
    """Split text or list of texts into words or characters.

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
        tuple (min_n, max_n), default=(1).
        The lower and upper boundary of the range of n-values for different n-grams to
        be extracted. All values of n such that min_n <= n <= max_n will be used.
    Examples
    --------
    >>> from lenlp import analyzer

    >>> analyzer.analyze("Hello, world!", analyzer="word")
    ['Hello,', 'world!']

    >>> analyzer.analyze("Hello, world!", analyzer="char_wb", ngram_range=(3, 3))
    ['Hel', 'ell', 'llo', 'lo,', 'o, ', ', w', ' wo', 'wor', 'orl', 'rld', 'ld!']

    >>> analyzer.analyze(["hello, world", "good"], analyzer="char", ngram_range=(2, 3))
    [['he', 'el', 'll', 'lo', 'o,', ', ', ' w', 'wo', 'or', 'rl', 'ld', 'hel', 'ell', 'llo', 'lo,', 'o, ', ', w', ' wo', 'wor', 'orl', 'rld'], ['go', 'oo', 'od', 'goo', 'ood']]

    """
    return_string = True if isinstance(x, str) else False
    x = [x] if isinstance(x, str) else x
    n_sizes = list(range(ngram_range[0], ngram_range[1] + 1))

    match analyzer:
        case "word":
            y = rssplit_words_many(x, n_sizes=n_sizes)
        case "char":
            y = rschar_ngrams_many(x, n_sizes=n_sizes)
        case "char_wb":
            y = rschar_wb_ngrams_many(x, n_sizes=n_sizes)

    return y[0] if return_string else y
