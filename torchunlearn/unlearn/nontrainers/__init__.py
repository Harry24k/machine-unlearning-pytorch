"""Non-training (closed-form / weight-edit / architectural) unlearning methods.

Import the concrete unlearners from :mod:`torchunlearn.unlearn` (preferred) or
directly from their modules, e.g. ``from torchunlearn.unlearn.nontrainers
.negmerge import NegMerge``.

This file exists so that ``setuptools.find_packages()`` treats this directory
as a package. Without it the subpackage is silently dropped from the built
wheel/sdist.
"""
