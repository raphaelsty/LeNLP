import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2

from .count_vectorizer import CountVectorizer


class TfidfVectorizer(CountVectorizer):
    """TfidfVectorizer is a class that converts a collection of text documents to a sparse
    tfidf matrix.

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

    >>> tfidf_vectorizer = sparse.TfidfVectorizer(
    ...     analyzer="word",
    ...     normalize=True,
    ...     stop_words=None,
    ... )

    >>> x = ["Hello, world!", "How are you?"]

    >>> tfidf_vectorizer = tfidf_vectorizer.fit(x)
    >>> matrix = tfidf_vectorizer.transform(x)
    >>> matrix.shape
    (2, 5)

    >>> len(tfidf_vectorizer.vocabulary)
    5

    >>> matrix = tfidf_vectorizer.fit_transform(x)
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
        super().__init__(
            analyzer=analyzer,
            ngram_range=ngram_range,
            normalize=normalize,
            stop_words=stop_words,
        )

        self.idf = None

    def fit(self, raw_documents: list[str]) -> None:
        matrix = super().fit_transform(raw_documents=raw_documents)
        self.update(matrix=matrix)
        return self

    def update(self, matrix: csr_matrix) -> csr_matrix:
        """Update the idf values."""
        tf = (matrix > 0).sum(axis=0)
        self.idf = (
            np.squeeze(a=np.asarray(a=np.log((matrix.shape[0] + 1.0) / (tf + 1.0)))) + 1
        )

    def _transform(self, matrix: csr_matrix) -> csr_matrix:
        """Transform a count matrix to a bm25 matrix."""
        matrix.data *= np.take(
            a=self.idf,
            indices=matrix.indices,
        )

        inplace_csr_row_normalize_l2(X=matrix)
        return matrix

    def transform(self, raw_documents: list[str]) -> csr_matrix:
        """Transform documents to document-term matrix."""
        values, row_indices, column_indices = self.sparse_matrix.transform(
            raw_documents
        )
        return self._transform(
            matrix=csr_matrix(
                arg1=(values, (row_indices, column_indices)),
                shape=(len(raw_documents), self.sparse_matrix.get_num_cols()),
                dtype=np.float32,
            )
        )

    def fit_transform(self, raw_documents: list[str]) -> csr_matrix:
        """Learn the vocabulary dictionary and return the CountVectorizer object."""
        values, row_indices, column_indices = self.sparse_matrix.fit_transform(
            raw_documents
        )

        matrix = csr_matrix(
            arg1=(values, (row_indices, column_indices)),
            shape=(len(raw_documents), self.sparse_matrix.get_num_cols()),
            dtype=np.float32,
        )

        self.update(matrix=matrix)

        return self._transform(
            matrix=matrix,
        )
