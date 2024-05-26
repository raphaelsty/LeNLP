import numpy as np
from scipy.sparse import csr_matrix
from sklearn.utils.sparsefuncs_fast import inplace_csr_row_normalize_l2

from .tfidf_vectorizer import TfidfVectorizer


class BM25Vectorizer(TfidfVectorizer):
    """BM25Vectorizer is a class that converts a collection of text documents to a sparse
    bm25 matrix.

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
    b
        The impact of document length normalization.  Default is `0.75`, Higher will
        penalize longer documents more.
    k1
        How quickly the impact of term frequency saturates.  Default is `1.5`, Higher
        will make term frequency more influential.
    epsilon
        Smoothing term. Default is `0`.

    Examples
    --------
    >>> from lenlp import sparse

    >>> bm25_vectorizer = sparse.BM25Vectorizer(
    ...     analyzer="word",
    ...     normalize=True,
    ...     stop_words=None,
    ... )

    >>> x = ["Hello, world!", "How are you?"]

    >>> bm25_vectorizer = bm25_vectorizer.fit(x)
    >>> matrix = bm25_vectorizer.transform(x)
    >>> matrix.shape
    (2, 5)

    >>> len(bm25_vectorizer.vocabulary)
    5

    >>> matrix = bm25_vectorizer.fit_transform(x)
    >>> matrix.shape
    (2, 5)

    """

    def __init__(
        self,
        analyzer: str = "word",
        ngram_range: tuple[int, int] = (1, 1),
        normalize: bool = True,
        stop_words: list[str] = None,
        k1: float = 1.5,
        b: float = 0.75,
        epsilon: float = 0,
    ) -> None:
        super().__init__(
            analyzer=analyzer,
            ngram_range=ngram_range,
            normalize=normalize,
            stop_words=stop_words,
        )

        self.k1 = k1
        self.b = b
        self.epsilon = epsilon
        self.average_len = None

    def update(self, matrix: csr_matrix) -> csr_matrix:
        """Update the idf values."""
        self.tf = (matrix > 0).sum(axis=0)
        len_documents = (matrix).sum(axis=1)
        self.average_len = len_documents.mean()
        self.count = matrix.shape[0]

        self.idf = np.squeeze(
            a=np.asarray(a=np.log((self.count - self.tf + 0.5) / (self.tf + 0.5) + 1))
        )

    def _transform(self, matrix: csr_matrix) -> csr_matrix:
        """Transform a count matrix to a bm25 matrix."""
        len_documents = (matrix).sum(axis=1)
        regularization = np.squeeze(
            a=np.asarray(
                a=(
                    self.k1 * (1 - self.b + self.b * (len_documents / self.average_len))
                ).flatten()
            )
        )

        numerator = matrix.copy()
        denominator = matrix.copy().tocsc()
        numerator.data = numerator.data * (self.k1 + 1)
        denominator.data += np.take(a=regularization, indices=denominator.indices)
        matrix.data = (numerator.data / denominator.tocsr().data) + self.epsilon

        matrix = matrix.multiply(other=self.idf).tocsr()
        inplace_csr_row_normalize_l2(X=matrix)
        return matrix
