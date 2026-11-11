from __future__ import annotations
from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
from typing import Any, Mapping
from cdtab.ast import Predicate
from cdtab.errors import ConstraintTypeError
from cdtab.types import FiniteType

@dataclass(frozen=True)
class Field:
    name: str
    field_type: FiniteType

    def __post_init__(self) -> None:
        if not self.name or not self.name.isascii():
            raise ConstraintTypeError('field name must be non-empty ASCII')
        if any((character in self.name for character in (',', '\n', '\r'))):
            raise ConstraintTypeError('field name contains a reserved separator')

@dataclass(frozen=True)
class Schema:
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        names = self.names
        if not names:
            raise ConstraintTypeError('schema must contain at least one field')
        if len(set(names)) != len(names):
            raise ConstraintTypeError('field names must be unique')

    @property
    def names(self) -> tuple[str, ...]:
        return tuple((field.name for field in self.fields))

    @property
    def types(self) -> Mapping[str, FiniteType]:
        return {field.name: field.field_type for field in self.fields}

    def field(self, name: str) -> Field:
        for field in self.fields:
            if field.name == name:
                return field
        raise ConstraintTypeError(f'unknown field {name!r}')

@dataclass(frozen=True)
class ConstraintSpec:
    schema: Schema
    constraint: Predicate
    order: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.order) != len(self.schema.names) or set(self.order) != set(self.schema.names):
            raise ConstraintTypeError('order must be a permutation of schema fields')
        self.constraint.validate(self.schema.types)
        undeclared = self.constraint.variables() - frozenset(self.schema.names)
        if undeclared:
            raise ConstraintTypeError(f'undeclared variables: {sorted(undeclared)}')
        for name in self.schema.names:
            if self.schema.field(name).field_type.cardinality <= 0:
                raise ConstraintTypeError(f'empty domain for {name!r}')

    def domain(self, name: str) -> tuple[int, ...]:
        return self.schema.field(name).field_type.values()

    def domain_size(self, name: str) -> int:
        return self.schema.field(name).field_type.cardinality

    def contains(self, name: str, value: int) -> bool:
        return self.schema.field(name).field_type.contains(value)

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(_canonical(self), sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(payload.encode('utf-8')).hexdigest()

def _canonical(value: Any) -> Any:
    if is_dataclass(value):
        return {'type': type(value).__name__, 'fields': {item.name: _canonical(getattr(value, item.name)) for item in fields(value)}}
    if isinstance(value, tuple):
        return [_canonical(item) for item in value]
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _canonical(item) for key, item in sorted(value.items())}
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(f'cannot canonicalize {type(value).__name__}')
