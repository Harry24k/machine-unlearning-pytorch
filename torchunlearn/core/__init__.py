"""Pure, side-effect-free building blocks (losses, regularizers, etc.).

Per the refactoring guide §2.1, modules in `core/` must not depend on
any other layer (engine, algorithms, interface). They are safe to import
from anywhere.
"""
from .losses import classification_loss, l1_parameter_penalty

__all__ = ["classification_loss", "l1_parameter_penalty"]
