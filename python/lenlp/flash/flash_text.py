from lenlp._rslenlp import RSKeywordProcessor

__all__ = ["FlashText"]


class FlashText:
    """FlashText retrieve keywords from text.

    Parameters
    ----------
    lowercase
        bool, default=True.
        Whether to lowercase the text before extracting keywords.
    normalize
        bool, default=True.
        Whether to normalize the text before extracting keywords. It will lowercase the text
        and remove punctuation.

    Examples
    --------
    >>> from lenlp import flash

    >>> flash_text = flash.FlashText(normalize=True)
    >>> flash_text = flash_text.add(["hello", "world"])

    >>> flash_text.extract(["Hello, world!", "world", "hello"])
    [[('hello', 0, 5), ('world', 7, 12)], [('world', 0, 5)], [('hello', 0, 5)]]

    """

    def __init__(self, lowercase: bool = True, normalize: bool = True) -> None:
        self.flash = RSKeywordProcessor(lowercase=lowercase, normalize=normalize)

    def add(self, x: str | list[str]) -> None:
        """Add a keyword to the FlashText object."""
        x = [x] if isinstance(x, str) else x
        self.flash.add_keywords_many(x)
        return self

    def extract(self, x: str | list[str]) -> list[str]:
        """Extract keywords from a sentence."""
        is_string = isinstance(x, str)
        x = [x] if isinstance(x, str) else x
        y = self.flash.extract_keywords_many(x)
        return y[0] if is_string else y
