from rslenlp import SparseMatrixBuilder
from scipy.sparse import csr_matrix

__all__ = ["CountVectorizer"]


class CountVectorizer:
    """CountVectorizer is a class that converts a collection of text documents to a sparse
    matrix.

    Parameters
    ----------
    analyzer
        {word, char, char_wb}, default=word.
        Whether the feature should be made of word n-gram or character n-grams. Option
        char_wb creates character n-grams only from text inside word boundaries;
        n-grams at the edges of words are padded with space.
    ngram_range
        tuple (min_n, max_n), default=(1, 1).
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
    >>> from lenlp import sparse

    >>> count_vectorizer = sparse.CountVectorizer(
    ...     analyzer="word",
    ...     normalize=True,
    ...     stop_words=None,
    ... )

    >>> x = ["Hello, world!", "How are you?"]

    >>> count_vectorizer = count_vectorizer.fit(x)

    >>> matrix = count_vectorizer.transform(x)
    >>> matrix.shape
    (2, 5)

    >>> matrix.toarray()
    array([[1, 1, 0, 0, 0],
           [0, 0, 1, 1, 1]], dtype=uint64)

    >>> len(count_vectorizer.vocabulary)
    5

    >>> matrix = count_vectorizer.fit_transform(x)
    >>> matrix.shape
    (2, 5)

    """

    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 1),
        normalize: bool = True,
        stop_words: list[str] = None,
    ) -> None:
        assert analyzer in ("word", "char", "char_wb")

        self.sparse_matrix = SparseMatrixBuilder(
            analyzer=analyzer,
            n_sizes=list(range(ngram_range[0], ngram_range[1] + 1)),
            normalize=normalize,
            stop_words=stop_words,
        )

        self.fitted = False

    @property
    def vocabulary(self) -> dict[str, int]:
        """Get the vocabulary of the CountVectorizer object."""
        return self.sparse_matrix.get_vocab()

    def fit(self, raw_documents: list[str]) -> None:
        """Learn the vocabulary dictionary and return the CountVectorizer object."""
        self.fitted = True
        self.sparse_matrix.fit(raw_documents)
        return self

    def transform(self, raw_documents: list[str]) -> csr_matrix:
        """Transform documents to document-term matrix."""
        if not self.fitted:
            raise ValueError("Call fit method before calling transform method.")

        values, row_indices, column_indices = self.sparse_matrix.transform(
            raw_documents
        )

        return csr_matrix(
            arg1=(values, (row_indices, column_indices)),
            shape=(len(raw_documents), self.sparse_matrix.get_num_cols()),
        )

    def fit_transform(self, raw_documents: list[str]) -> csr_matrix:
        """Learn the vocabulary dictionary and return the CountVectorizer object."""
        self.fitted = True

        values, row_indices, column_indices = self.sparse_matrix.fit_transform(
            raw_documents
        )

        return csr_matrix(
            arg1=(values, (row_indices, column_indices)),
            shape=(len(raw_documents), self.sparse_matrix.get_num_cols()),
        )
