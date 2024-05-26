import setuptools
from setuptools_rust import Binding, RustExtension

from lenlp.__version__ import __version__

base_packages = ["scikit-learn  >= 1.5.0", "scipy >= 1.13.1"]
dev = ["maturin >= 1.5.1", "pytest-cov >= 5.0.0", "pytest >= 7.4.4", "ruff >= 0.1.15"]


setuptools.setup(
    name="lenlp",
    version=f"{__version__}",
    author="Raphael Sourty",
    author_email="raphael.sourty@gmail.com",
    long_description_content_type="text/markdown",
    url="https://github.com/raphaelsty/lenlp",
    download_url="https://github.com/raphaelsty/lenlp/archive/v_01.tar.gz",
    keywords=[],
    packages=setuptools.find_packages(),
    install_requires=base_packages,
    extras_require={"dev": base_packages + dev},
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    rust_extensions=[RustExtension("rslenlp", binding=Binding.PyO3)],
    setup_requires=["setuptools-rust>=1.9.0"],
)
