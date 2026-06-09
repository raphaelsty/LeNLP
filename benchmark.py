"""Benchmark every LeNLP algorithm on CPU.

Usage: python benchmark.py [output.json]

Times each algorithm on the scifact corpus (5183 documents) and dumps the
results as JSON so two runs can be compared.
"""

import json
import sys
import time

from lenlp import analyzer, counter, flash, normalizer, sparse


def load_corpus(path: str = "evaluation_datasets/scifact/corpus.jsonl") -> list[str]:
    documents = []
    with open(path) as f:
        for line in f:
            doc = json.loads(line)
            documents.append(f"{doc['title']} {doc['text']}")
    return documents


def timeit(fn, repeat: int = 5) -> tuple[float, object]:
    """Return the best wall-clock time over `repeat` runs and the last result."""
    best = float("inf")
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def main() -> None:
    documents = load_corpus()
    print(f"{len(documents)} documents, {sum(len(d) for d in documents) / 1e6:.1f}M chars")

    timings: dict[str, float] = {}
    checksums: dict[str, object] = {}

    # ------------------------------------------------------------------
    # Normalizer
    # ------------------------------------------------------------------
    t, out = timeit(lambda: normalizer.normalize(documents))
    timings["normalizer.normalize"] = t
    checksums["normalizer.normalize"] = [len(out), sum(len(s) for s in out), out[42][:80]]

    # ------------------------------------------------------------------
    # Analyzer
    # ------------------------------------------------------------------
    for name, kwargs in [
        ("word (1,1)", dict(analyzer="word", ngram_range=(1, 1))),
        ("word (1,2)", dict(analyzer="word", ngram_range=(1, 2))),
        ("char (3,5)", dict(analyzer="char", ngram_range=(3, 5))),
        ("char_wb (3,5)", dict(analyzer="char_wb", ngram_range=(3, 5))),
    ]:
        t, out = timeit(lambda kw=kwargs: analyzer.analyze(documents, **kw), repeat=3)
        timings[f"analyzer.analyze {name}"] = t
        checksums[f"analyzer.analyze {name}"] = [
            sum(len(x) for x in out),
            out[42][:5],
        ]

    # ------------------------------------------------------------------
    # Counter
    # ------------------------------------------------------------------
    for name, kwargs in [
        ("word (1,1)", dict(analyzer="word", ngram_range=(1, 1))),
        ("char (3,5)", dict(analyzer="char", ngram_range=(3, 5))),
    ]:
        t, out = timeit(lambda kw=kwargs: counter.count(documents, **kw), repeat=3)
        timings[f"counter.count {name}"] = t
        checksums[f"counter.count {name}"] = [
            sum(len(d) for d in out),
            sum(sum(d.values()) for d in out),
        ]

    # ------------------------------------------------------------------
    # Sparse vectorizers
    # ------------------------------------------------------------------
    for cls_name, cls in [
        ("CountVectorizer", sparse.CountVectorizer),
        ("TfidfVectorizer", sparse.TfidfVectorizer),
        ("BM25Vectorizer", sparse.BM25Vectorizer),
    ]:
        for an_name, kwargs in [
            ("word (1,1)", dict(analyzer="word", ngram_range=(1, 1))),
            ("char (3,5)", dict(analyzer="char", ngram_range=(3, 5))),
        ]:
            t, out = timeit(
                lambda kw=kwargs, c=cls: c(**kw).fit_transform(documents), repeat=3
            )
            key = f"{cls_name}.fit_transform {an_name}"
            timings[key] = t
            checksums[key] = [out.shape, out.nnz, round(float(out.sum()), 2)]

            vec = cls(**kwargs)
            vec.fit(documents)
            t, out = timeit(lambda v=vec: v.transform(documents), repeat=3)
            key = f"{cls_name}.transform {an_name}"
            timings[key] = t
            checksums[key] = [out.shape, out.nnz, round(float(out.sum()), 2)]

    # ------------------------------------------------------------------
    # FlashText
    # ------------------------------------------------------------------
    # Build a realistic keyword set: the 5000 most frequent normalized words.
    freq = counter.count(" ".join(documents))
    keywords = sorted(freq, key=freq.get, reverse=True)[:5000]

    def build_flash():
        ft = flash.FlashText(normalize=True)
        ft.add(keywords)
        return ft

    t, ft = timeit(build_flash, repeat=3)
    timings["flash.add 5k keywords"] = t

    t, out = timeit(lambda: ft.extract(documents[:1000]), repeat=3)
    timings["flash.extract 1k docs"] = t
    checksums["flash.extract 1k docs"] = [sum(len(x) for x in out)]

    print()
    for key, value in timings.items():
        print(f"{key:55s} {value * 1000:10.1f} ms")

    if len(sys.argv) > 1:
        with open(sys.argv[1], "w") as f:
            json.dump({"timings": timings, "checksums": {k: repr(v) for k, v in checksums.items()}}, f, indent=2)
        print(f"\nsaved to {sys.argv[1]}")


if __name__ == "__main__":
    main()
