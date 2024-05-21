from setuptools import setup
from setuptools_rust import RustExtension

setup(
    name="antelope",
    version="0.1.0",
    packages=["antelope"],
    rust_extensions=[
        RustExtension("rsantelope.rsantelope", "Cargo.toml", binding="pyo3")
    ],
    include_package_data=True,
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Rust",
        "Programming Language :: Python :: 3",
    ],
)
