"""Public API for exact constraint-aware native-token decoding."""

from cdtab.ast import (
    Add,
    And,
    Const,
    Eq,
    Ge,
    Gt,
    Le,
    Lt,
    MulBounded,
    MulConst,
    RoundHalfEvenDiv,
    Sub,
    Var,
)
from cdtab.logits import apply_exact_mask
from cdtab.spec import ConstraintSpec, Field, Schema
from cdtab.types import BoundedInt, FiniteEnum, FixedPoint

from .backends import ExtensionLimits, compile_constraints
from .compiler import CompileLimits
from .frontend import Category, ExtendedSpec, IsValue, Or
from .order import OrderPolicy
from .runtime import make_masker
from .token_contract import GPT2ByteBPEAdapter, StaticTokenAdapter

__all__ = [
    "Add",
    "And",
    "BoundedInt",
    "Category",
    "CompileLimits",
    "Const",
    "ConstraintSpec",
    "Eq",
    "ExtendedSpec",
    "ExtensionLimits",
    "Field",
    "FiniteEnum",
    "FixedPoint",
    "GPT2ByteBPEAdapter",
    "Ge",
    "Gt",
    "IsValue",
    "Le",
    "Lt",
    "MulBounded",
    "MulConst",
    "Or",
    "OrderPolicy",
    "RoundHalfEvenDiv",
    "Schema",
    "StaticTokenAdapter",
    "Sub",
    "Var",
    "apply_exact_mask",
    "compile_constraints",
    "make_masker",
]
