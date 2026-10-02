from dataclasses import dataclass
from typing import Any, Optional, Sequence


@dataclass
class HParam:
    """Specification for a single hyperparameter of an unlearning algorithm."""
    name: str
    type: type
    default: Any = None
    required: bool = False
    description: str = ""
    choices: Optional[Sequence] = None
    range: Optional[tuple] = None
    example: Any = None
    category: str = "hparam"   # "hparam" or "setup"

    def validate(self, value):
        if value is None:
            if self.required:
                raise ValueError(f"'{self.name}' is required")
            return self.default
        if self.type is float and isinstance(value, int):
            value = float(value)
        elif self.type is list and isinstance(value, tuple):
            value = list(value)
        elif not isinstance(value, self.type):
            raise TypeError(
                f"'{self.name}' must be {self.type.__name__}, "
                f"got {type(value).__name__}"
            )
        if self.choices is not None and value not in self.choices:
            raise ValueError(f"'{self.name}' must be one of {list(self.choices)}")
        if self.range is not None and not (self.range[0] <= value <= self.range[1]):
            raise ValueError(f"'{self.name}' must be in range {self.range}")
        return value