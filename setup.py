import setuptools
from setuptools_rust import RustExtension

from lenlp import __version__

with open(file="README.md", mode="r", encoding="utf-8") as fh:
    long_description = fh.read()

base_packages = ["scikit-learn >= 1.5.0", "scipy >= 1.13.0"]

setuptools.setup(
    name="lenlp",
    version=f"{__version__}",
    license="MIT",
    authors="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    long_description=long_description,
    long_description_content_type="text/markdown",
    rust_extensions=[RustExtension("rslenlp.rslenlp", "Cargo.toml", binding="pyo3")],
    url="https://github.com/raphaelsty/lenlp",
    download_url="https://github.com/user/lenlp/archive/v_01.tar.gz",
    keywords=[
        "information retrieval",
        "nlp",
        "rust",
        "bm25",
        "tfidf",
        "flashtext",
    ],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.6",
)
