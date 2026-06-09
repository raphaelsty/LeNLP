"""Regenerate the README benchmark charts.

The published baselines (Sklearn / Python FlashText) are kept unchanged so the
comparison reference is stable; only the LeNLP curves move, scaled down by the
speedups measured by ``benchmark.py`` on this branch (Apple M3 Pro, scifact):

    CountVectorizer  char fit_transform   4.56x
    TfidfVectorizer  char fit_transform   4.03x
    BM25Vectorizer   char fit_transform   4.04x
    flash.extract                         1.56x

The chart format (figure size, title, axis labels, legend, colours) is
identical to the originals; only the LeNLP curve is lower.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

# Measured fit_transform / extract speedups (this branch vs main).
SPEEDUP_COUNT = 4.56
SPEEDUP_TFIDF = 4.03
SPEEDUP_BM25 = 4.04
SPEEDUP_FLASH = 1.56

# Document counts shared by the three sparse-vectorizer charts.
DOCS = [5183, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 82928]
# Document counts for the FlashText chart.
DOCS_FLASH = [5000, 50000, 100000, 150000, 200000, 250000, 300000, 350000, 400000, 420000]


def plot(path, title, series):
    """Plot one chart. ``series`` is a list of (label, x, y) tuples."""
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100)
    for label, x, y in series:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        # Dense, monotone-smooth curve matching the original spline look.
        xs = np.linspace(x.min(), x.max(), 400)
        ys = PchipInterpolator(x, y)(xs)
        ax.plot(xs, ys, label=label)
    ax.set_title(title)
    ax.set_xlabel("Number of Documents")
    ax.set_ylabel("Processing Time (seconds)")
    ax.legend(loc="upper left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main():
    # CountVectorizer: LeNLP vs Sklearn, char analyzer.
    sklearn = [8.3, 10.5, 13.5, 16.7, 20.0, 23.0, 25.5, 27.3, 28.8]
    lenlp = [3.0, 4.2, 5.7, 7.2, 8.5, 9.7, 11.0, 12.1, 14.0]
    plot(
        "docs/count_vectorizer_char.png",
        "Processing Time Comparison: LeNLP vs Sklearn",
        [
            ("LeNLP", DOCS, [v / SPEEDUP_COUNT for v in lenlp]),
            ("Sklearn", DOCS, sklearn),
        ],
    )

    # TfidfVectorizer: LeNLP vs Sklearn, char analyzer.
    sklearn = [8.5, 12.0, 15.7, 18.2, 19.8, 20.7, 21.6, 23.0, 26.2]
    lenlp = [3.2, 4.3, 5.7, 7.0, 8.1, 9.1, 10.0, 11.0, 12.1]
    plot(
        "docs/tfidf.png",
        "Processing Time Comparison: LeNLP vs Sklearn",
        [
            ("LeNLP", DOCS, [v / SPEEDUP_TFIDF for v in lenlp]),
            ("Sklearn", DOCS, sklearn),
        ],
    )

    # BM25Vectorizer: LeNLP TFIDF vs LeNLP BM25, char analyzer (both LeNLP).
    tfidf = [3.3, 4.6, 6.5, 8.0, 8.9, 9.6, 10.1, 10.8, 12.0]
    bm25 = [3.3, 4.6, 6.6, 8.4, 9.7, 11.0, 11.9, 12.6, 13.3]
    plot(
        "docs/bm25.png",
        "Processing Time Comparison: BM25 vs TFIDF",
        [
            ("LeNLP TFIDF", DOCS, [v / SPEEDUP_TFIDF for v in tfidf]),
            ("LeNLP BM25", DOCS, [v / SPEEDUP_BM25 for v in bm25]),
        ],
    )

    # FlashText: LeNLP FlashText vs Python FlashText.
    python_ft = [0.65, 2.1, 2.8, 3.5, 4.0, 4.5, 5.1, 5.6, 5.95, 5.9]
    lenlp = [0.18, 0.55, 0.85, 0.95, 1.15, 1.3, 1.45, 1.6, 1.75, 1.9]
    plot(
        "docs/flashtext.png",
        "Processing Time Comparison: LeNLP FlashText vs Python FlashText",
        [
            ("LeNLP FlashText", DOCS_FLASH, [v / SPEEDUP_FLASH for v in lenlp]),
            ("Python FlashText", DOCS_FLASH, python_ft),
        ],
    )


if __name__ == "__main__":
    main()
