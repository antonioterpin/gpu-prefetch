"""Setup script for the GPU prefetching package."""

import pathlib

import setuptools

setuptools.setup(
    name="gpuprefetch",
    version="0.1.0",
    author="Antonio Terpin",
    author_email="aterpin@ethz.ch",
    description="Package for data prefetching on GPU.",
    url="http://github.com/antonioterpin/gpu-prefetch",
    long_description=pathlib.Path("README.md").read_text(),
    long_description_content_type="text/markdown",
    package_dir={"": "src"},
    packages=["gpuprefetch"],
    include_package_data=True,
    install_requires=[
        "cupy-cuda12x",
    ],
    extras_require={
        "dev": ["pytest", "setuptools"],
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
    ],
)
