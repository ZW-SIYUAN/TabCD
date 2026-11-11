from __future__ import annotations
from dataclasses import dataclass
from typing import Protocol
from cdtab.errors import ConstraintTypeError

class FiniteType(Protocol):

    @property
    def scale(self) -> int:
        ...

    def values(self) -> tuple[int, ...]:
        ...

    @property
    def cardinality(self) -> int:
        ...

    def contains(self, value: int) -> bool:
        ...

@dataclass(frozen=True)
class BoundedInt:
    minimum: int
    maximum: int
    scale: int = 0

    def __post_init__(self) -> None:
        if self.minimum > self.maximum:
            raise ConstraintTypeError('minimum exceeds maximum')
        if self.scale != 0:
            raise ConstraintTypeError('BoundedInt scale must be zero')

    def values(self) -> tuple[int, ...]:
        return tuple(range(self.minimum, self.maximum + 1))

    @property
    def cardinality(self) -> int:
        return self.maximum - self.minimum + 1

    def contains(self, value: int) -> bool:
        return self.minimum <= int(value) <= self.maximum

@dataclass(frozen=True)
class FixedPoint:
    minimum_scaled: int
    maximum_scaled: int
    scale: int

    def __post_init__(self) -> None:
        if self.minimum_scaled > self.maximum_scaled:
            raise ConstraintTypeError('minimum_scaled exceeds maximum_scaled')
        if self.scale < 0:
            raise ConstraintTypeError('fixed-point scale must be non-negative')

    def values(self) -> tuple[int, ...]:
        return tuple(range(self.minimum_scaled, self.maximum_scaled + 1))

    @property
    def cardinality(self) -> int:
        return self.maximum_scaled - self.minimum_scaled + 1

    def contains(self, value: int) -> bool:
        return self.minimum_scaled <= int(value) <= self.maximum_scaled

@dataclass(frozen=True)
class FiniteEnum:
    members: tuple[int, ...]
    scale: int = 0

    def __post_init__(self) -> None:
        if self.scale < 0:
            raise ConstraintTypeError('enum scale must be non-negative')
        if not self.members:
            raise ConstraintTypeError('enum must contain at least one member')
        if len(set(self.members)) != len(self.members):
            raise ConstraintTypeError('enum members must be unique')

    def values(self) -> tuple[int, ...]:
        return tuple(sorted(self.members))

    @property
    def cardinality(self) -> int:
        return len(self.members)

    def contains(self, value: int) -> bool:
        return int(value) in self.members
