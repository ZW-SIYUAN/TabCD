from __future__ import annotations
from dataclasses import dataclass
from math import prod
from typing import Mapping, Protocol
from cdtab.errors import ConstraintTypeError
from cdtab.types import FiniteType
SchemaTypes = Mapping[str, FiniteType]

class Expr(Protocol):

    def evaluate(self, env: Mapping[str, int]) -> int:
        ...

    def variables(self) -> frozenset[str]:
        ...

    def infer_scale(self, schema: SchemaTypes) -> int:
        ...

@dataclass(frozen=True)
class Var:
    name: str

    def evaluate(self, env: Mapping[str, int]) -> int:
        return int(env[self.name])

    def variables(self) -> frozenset[str]:
        return frozenset((self.name,))

    def infer_scale(self, schema: SchemaTypes) -> int:
        if self.name not in schema:
            raise ConstraintTypeError(f'undeclared variable {self.name!r}')
        return int(schema[self.name].scale)

@dataclass(frozen=True)
class Const:
    value: int
    scale: int = 0

    def evaluate(self, env: Mapping[str, int]) -> int:
        del env
        return self.value

    def variables(self) -> frozenset[str]:
        return frozenset()

    def infer_scale(self, schema: SchemaTypes) -> int:
        del schema
        if self.scale < 0:
            raise ConstraintTypeError('constant scale must be non-negative')
        return self.scale

def _same_scale(left: Expr, right: Expr, schema: SchemaTypes, op: str) -> int:
    left_scale = left.infer_scale(schema)
    right_scale = right.infer_scale(schema)
    if left_scale != right_scale:
        raise ConstraintTypeError(f'{op} scale mismatch {left_scale}!={right_scale}')
    return left_scale

@dataclass(frozen=True)
class Add:
    left: Expr
    right: Expr

    def evaluate(self, env: Mapping[str, int]) -> int:
        return self.left.evaluate(env) + self.right.evaluate(env)

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()

    def infer_scale(self, schema: SchemaTypes) -> int:
        return _same_scale(self.left, self.right, schema, 'add')

@dataclass(frozen=True)
class Sub:
    left: Expr
    right: Expr

    def evaluate(self, env: Mapping[str, int]) -> int:
        return self.left.evaluate(env) - self.right.evaluate(env)

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()

    def infer_scale(self, schema: SchemaTypes) -> int:
        return _same_scale(self.left, self.right, schema, 'subtract')

@dataclass(frozen=True)
class MulConst:
    expr: Expr
    coefficient: int
    coefficient_scale: int = 0

    def evaluate(self, env: Mapping[str, int]) -> int:
        return self.expr.evaluate(env) * self.coefficient

    def variables(self) -> frozenset[str]:
        return self.expr.variables()

    def infer_scale(self, schema: SchemaTypes) -> int:
        if self.coefficient_scale < 0:
            raise ConstraintTypeError('coefficient scale must be non-negative')
        return self.expr.infer_scale(schema) + self.coefficient_scale

@dataclass(frozen=True)
class MulBounded:
    left: Expr
    right: Expr

    def evaluate(self, env: Mapping[str, int]) -> int:
        return self.left.evaluate(env) * self.right.evaluate(env)

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()

    def infer_scale(self, schema: SchemaTypes) -> int:
        return self.left.infer_scale(schema) + self.right.infer_scale(schema)

def round_half_even_ratio(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ConstraintTypeError('rounding denominator must be positive')
    sign = -1 if numerator < 0 else 1
    quotient, remainder = divmod(abs(numerator), denominator)
    doubled = 2 * remainder
    if doubled < denominator:
        rounded = quotient
    elif doubled > denominator:
        rounded = quotient + 1
    else:
        rounded = quotient if quotient % 2 == 0 else quotient + 1
    return sign * rounded

@dataclass(frozen=True)
class RoundHalfEvenDiv:
    expr: Expr
    divisor: int
    output_scale: int

    def evaluate(self, env: Mapping[str, int]) -> int:
        return round_half_even_ratio(self.expr.evaluate(env), self.divisor)

    def variables(self) -> frozenset[str]:
        return self.expr.variables()

    def infer_scale(self, schema: SchemaTypes) -> int:
        input_scale = self.expr.infer_scale(schema)
        if input_scale < self.output_scale:
            raise ConstraintTypeError('rounding cannot increase scale')
        expected = 10 ** (input_scale - self.output_scale)
        if self.divisor != expected:
            raise ConstraintTypeError(f'rounding divisor {self.divisor} inconsistent with scales {input_scale}->{self.output_scale}; expected {expected}')
        return self.output_scale

class Predicate(Protocol):

    def evaluate(self, env: Mapping[str, int]) -> bool:
        ...

    def variables(self) -> frozenset[str]:
        ...

    def validate(self, schema: SchemaTypes) -> None:
        ...

@dataclass(frozen=True)
class _Comparison:
    left: Expr
    right: Expr

    def variables(self) -> frozenset[str]:
        return self.left.variables() | self.right.variables()

    def validate(self, schema: SchemaTypes) -> None:
        _same_scale(self.left, self.right, schema, type(self).__name__.lower())

@dataclass(frozen=True)
class Eq(_Comparison):

    def evaluate(self, env: Mapping[str, int]) -> bool:
        return self.left.evaluate(env) == self.right.evaluate(env)

@dataclass(frozen=True)
class Le(_Comparison):

    def evaluate(self, env: Mapping[str, int]) -> bool:
        return self.left.evaluate(env) <= self.right.evaluate(env)

@dataclass(frozen=True)
class Lt(_Comparison):

    def evaluate(self, env: Mapping[str, int]) -> bool:
        return self.left.evaluate(env) < self.right.evaluate(env)

@dataclass(frozen=True)
class Ge(_Comparison):

    def evaluate(self, env: Mapping[str, int]) -> bool:
        return self.left.evaluate(env) >= self.right.evaluate(env)

@dataclass(frozen=True)
class Gt(_Comparison):

    def evaluate(self, env: Mapping[str, int]) -> bool:
        return self.left.evaluate(env) > self.right.evaluate(env)

@dataclass(frozen=True)
class And:
    predicates: tuple[Predicate, ...]

    def __post_init__(self) -> None:
        if not self.predicates:
            raise ConstraintTypeError('And requires at least one predicate')

    def evaluate(self, env: Mapping[str, int]) -> bool:
        return all((predicate.evaluate(env) for predicate in self.predicates))

    def variables(self) -> frozenset[str]:
        return frozenset().union(*(predicate.variables() for predicate in self.predicates))

    def validate(self, schema: SchemaTypes) -> None:
        for predicate in self.predicates:
            predicate.validate(schema)
Comparison = Eq | Le | Lt | Ge | Gt

def atomic_predicates(predicate: Predicate) -> tuple[Comparison, ...]:
    if isinstance(predicate, And):
        atoms: list[Comparison] = []
        for child in predicate.predicates:
            atoms.extend(atomic_predicates(child))
        return tuple(atoms)
    if isinstance(predicate, (Eq, Le, Lt, Ge, Gt)):
        return (predicate,)
    raise TypeError(f'unsupported predicate {type(predicate).__name__}')

def contains_bounded_product(expr: Expr) -> bool:
    if isinstance(expr, MulBounded):
        return True
    if isinstance(expr, (Add, Sub)):
        return contains_bounded_product(expr.left) or contains_bounded_product(expr.right)
    if isinstance(expr, MulConst):
        return contains_bounded_product(expr.expr)
    if isinstance(expr, RoundHalfEvenDiv):
        return contains_bounded_product(expr.expr)
    return False

def evaluate_product(expressions: tuple[Expr, ...], env: Mapping[str, int]) -> int:
    return prod((expr.evaluate(env) for expr in expressions))
