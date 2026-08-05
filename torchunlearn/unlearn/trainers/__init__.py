"""Training-based unlearning algorithms.

Import the concrete trainers from :mod:`torchunlearn.unlearn` (preferred) or
directly from their modules, e.g. ``from torchunlearn.unlearn.trainers.finetune
import Finetune``.

This file exists so that ``setuptools.find_packages()`` treats this directory
as a package. Without it the subpackage is silently dropped from the built
wheel/sdist.
"""
