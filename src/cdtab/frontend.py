"""Finite typed schemas, Boolean rules, and semantic value encoding."""

import hashlib
import json
from dataclasses import dataclass

from cdtab.ast import And, Eq, Ge, Gt, Le, Lt
from cdtab.domains import IntSet
from cdtab.errors import ConstraintTypeError
from cdtab.spec import Field, Schema, _canonical
from cdtab.types import BoundedInt, FiniteEnum, FixedPoint

NUMERIC = BoundedInt, FixedPoint, FiniteEnum
COMPARISONS = Eq, Le, Lt, Ge, Gt


@dataclass(frozen=True)
class Category:
    labels: tuple[str, ...]
    scale: int = 0

    def __post_init__(self):
        if type(self.labels) is not tuple or not self.labels or self.scale != 0:
            raise ConstraintTypeError("categories require an immutable nonempty label tuple")
        if len(set(self.labels)) != len(self.labels):
            raise ConstraintTypeError("category labels must be unique")
        for value in self.labels:
            if (
                type(value) is not str
                or not value
                or (not value.isascii())
                or (value != value.strip())
                or any(c in value for c in ",\r\n")
                or (" is " in value)
                or (value in ("None", "placeholder"))
                or any(ord(c) < 32 or ord(c) == 127 for c in value)
            ):
                raise ConstraintTypeError("category label conflicts with native GReaT grammar")

    @property
    def cardinality(self):
        return len(self.labels)


@dataclass(frozen=True)
class Nullable:
    base: object

    def __post_init__(self):
        if type(self.base) not in NUMERIC + (Category,):
            raise ConstraintTypeError("nullable base must be a registered finite type")

    @property
    def scale(self):
        return self.base.scale

    @property
    def cardinality(self):
        return self.base.cardinality + 1


def base_type(kind):
    return kind.base if type(kind) is Nullable else kind


def plain_domain(kind):
    kind = base_type(kind)
    if type(kind) is Category:
        return IntSet.span(0, len(kind.labels) - 1)
    if type(kind) is FiniteEnum:
        return IntSet.points(kind.members)
    if type(kind) is BoundedInt:
        return IntSet.span(kind.minimum, kind.maximum)
    if type(kind) is FixedPoint:
        return IntSet.span(kind.minimum_scaled, kind.maximum_scaled)
    raise ConstraintTypeError("unregistered field type")


def null_code(kind):
    if type(kind) is not Nullable:
        raise ConstraintTypeError("field is not nullable")
    return plain_domain(kind).minimum - 1


def field_domain(kind):
    domain = plain_domain(kind)
    if type(kind) is Nullable:
        return IntSet(domain.intervals + ((null_code(kind), null_code(kind)),))
    return domain


def encode_value(kind, value):
    if value is None:
        return null_code(kind)
    base = base_type(kind)
    if type(base) is Category:
        if type(value) is not str or value not in base.labels:
            raise ConstraintTypeError("unknown categorical value")
        return base.labels.index(value)
    if type(value) is not int or not plain_domain(kind).contains(value):
        raise ConstraintTypeError("value must be an exact in-domain scaled integer")
    return value


def decode_value(kind, code):
    if type(code) is not int or not field_domain(kind).contains(code):
        raise ConstraintTypeError("invalid semantic value code")
    if type(kind) is Nullable and code == null_code(kind):
        return None
    base = base_type(kind)
    return base.labels[code] if type(base) is Category else code


@dataclass(frozen=True)
class IsValue:
    field: str
    value: object

    def variables(self):
        return frozenset((self.field,))


@dataclass(frozen=True)
class IsMissing:
    field: str

    def variables(self):
        return frozenset((self.field,))


@dataclass(frozen=True)
class Or:
    predicates: tuple

    def __post_init__(self):
        if type(self.predicates) is not tuple or not self.predicates:
            raise ConstraintTypeError("Or requires an immutable nonempty tuple")

    def variables(self):
        return frozenset().union(*(p.variables() for p in self.predicates))


@dataclass(frozen=True)
class Not:
    predicate: object

    def variables(self):
        return self.predicate.variables()


@dataclass(frozen=True)
class Implies:
    condition: object
    consequence: object

    def variables(self):
        return self.condition.variables() | self.consequence.variables()


@dataclass(frozen=True)
class IfElse:
    condition: object
    when_true: object
    when_false: object

    def variables(self):
        return self.condition.variables() | self.when_true.variables() | self.when_false.variables()


@dataclass(frozen=True)
class ExtendedSpec:
    schema: Schema
    constraint: object
    order: tuple

    def __post_init__(self):
        if type(self.schema) is not Schema or type(self.schema.fields) is not tuple:
            raise ConstraintTypeError("registered immutable schema required")
        if (
            type(self.order) is not tuple
            or len(self.order) != len(self.schema.names)
            or set(self.order) != set(self.schema.names)
        ):
            raise ConstraintTypeError("order must be an immutable field permutation")
        for field in self.schema.fields:
            if type(field) is not Field:
                raise ConstraintTypeError("unregistered field")
            field_domain(field.field_type)

    @property
    def fingerprint(self):
        data = json.dumps(_canonical(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()

    def evaluate(self, env):
        return evaluate(self.constraint, env, self.schema.types)


def evaluate(predicate, env, types):
    if type(predicate) in COMPARISONS:
        if any(type(base_type(types[n])) is Category for n in predicate.variables()):
            raise ConstraintTypeError("categorical fields require IsValue, not arithmetic")
        if any(
            type(types[n]) is Nullable and env[n] == null_code(types[n])
            for n in predicate.variables()
        ):
            return False
        return predicate.evaluate(env)
    if type(predicate) is IsValue:
        return env[predicate.field] == encode_value(types[predicate.field], predicate.value)
    if type(predicate) is IsMissing:
        return env[predicate.field] == null_code(types[predicate.field])
    if type(predicate) is And:
        return all(evaluate(p, env, types) for p in predicate.predicates)
    if type(predicate) is Or:
        return any(evaluate(p, env, types) for p in predicate.predicates)
    if type(predicate) is Not:
        return not evaluate(predicate.predicate, env, types)
    if type(predicate) is Implies:
        return not evaluate(predicate.condition, env, types) or evaluate(
            predicate.consequence, env, types
        )
    if type(predicate) is IfElse:
        chosen = (
            predicate.when_true
            if evaluate(predicate.condition, env, types)
            else predicate.when_false
        )
        return evaluate(chosen, env, types)
    raise ConstraintTypeError("unregistered Boolean predicate")
