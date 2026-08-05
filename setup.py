import re
from os import path

import setuptools

this_directory = path.abspath(path.dirname(__file__))

with open(path.join(this_directory, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

# Single source of truth for the version: torchunlearn/__init__.py
with open(path.join(this_directory, "torchunlearn", "__init__.py"), encoding="utf-8") as f:
    version = re.search(r'^__version__\s*=\s*["\'](.+?)["\']', f.read(), re.M).group(1)

setuptools.setup(
    name="torchunlearn",
    version=version,
    description=(
        "Torchunlearn is a PyTorch library that provides machine unlearning "
        "methods to remove the influence of forget data."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Harry Kim",
    author_email="24k.harry@gmail.com",
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    keywords=[
        "unlearning", "machine unlearning", "mu", "pytorch", "torch",
        "finetune", "neggrad", "randomlabel", "influence", "fisherforget",
        "salun", "scrub", "negmerge", "uam",
    ],
    # All of these are imported at `import torchunlearn` time:
    #   rm.py -> matplotlib;  _vis.py -> pandas, matplotlib
    #   metrics -> utils.mia -> scikit-learn
    #   nontrainers/* -> tqdm;  attacks/* -> scipy
    install_requires=[
        "torch>=1.7.1",
        "torchvision>=0.8.2",
        "numpy>=1.19.4",
        "scipy",
        "scikit-learn",
        "pandas",
        "matplotlib",
        "tqdm",
    ],
    python_requires=">=3.8",
    zip_safe=False,
    license="MIT",
    url="https://github.com/Harry24k/machine-unlearning-pytorch",
    project_urls={
        "Source": "https://github.com/Harry24k/machine-unlearning-pytorch",
        "Bug Tracker": "https://github.com/Harry24k/machine-unlearning-pytorch/issues",
        "Paper": "https://neurips.cc/virtual/2025/poster/116406",
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    include_package_data=True,
)
