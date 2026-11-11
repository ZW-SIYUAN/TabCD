from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from typing import Iterable
from cdtab.order_errors import OrderPolicyError, OrderSelectionError

class OrderMode(str, Enum):
    FIXED = 'fixed'
    PRESELECTED = 'preselected'
    DYNAMIC = 'dynamic'

def _normalize_order(order: Iterable[str]) -> tuple[str, ...]:
    return tuple((str(name) for name in order))

@dataclass(frozen=True)
class OrderPolicy:
    mode: OrderMode
    orders: tuple[tuple[str, ...], ...] = ()

    def __post_init__(self) -> None:
        try:
            mode = OrderMode(self.mode)
        except ValueError as error:
            raise OrderPolicyError(f'unknown order mode {self.mode!r}') from error
        normalized = tuple((tuple(order) for order in self.orders))
        if any((any((type(name) is not str for name in order)) for order in normalized)):
            raise OrderPolicyError('field names in order policies must be strings')
        object.__setattr__(self, 'mode', mode)
        object.__setattr__(self, 'orders', normalized)

    @classmethod
    def fixed(cls, order: Iterable[str]) -> 'OrderPolicy':
        return cls(OrderMode.FIXED, (_normalize_order(order),))

    @classmethod
    def preselected(cls, orders: Iterable[Iterable[str]]) -> 'OrderPolicy':
        return cls(OrderMode.PRESELECTED, tuple((_normalize_order(order) for order in orders)))

    @classmethod
    def dynamic(cls, allowed_orders: Iterable[Iterable[str]] | None=None) -> 'OrderPolicy':
        orders = () if allowed_orders is None else tuple((_normalize_order(order) for order in allowed_orders))
        return cls(OrderMode.DYNAMIC, orders)

    def validate(self, fields: Iterable[str]) -> None:
        names = tuple(fields)
        if not names or len(set(names)) != len(names):
            raise OrderPolicyError('schema fields must be non-empty and unique')
        if self.mode is OrderMode.FIXED and len(self.orders) != 1:
            raise OrderPolicyError('fixed policy requires exactly one order')
        if self.mode is OrderMode.PRESELECTED and (not self.orders):
            raise OrderPolicyError('preselected policy requires at least one order')
        normalized = set()
        for order in self.orders:
            if len(order) != len(names) or set(order) != set(names):
                raise OrderPolicyError('every declared order must be a permutation of schema fields')
            if order in normalized:
                raise OrderPolicyError('declared orders must be unique')
            normalized.add(order)

    def resolve_selection(self, fields: Iterable[str], selected_order: Iterable[str] | None) -> tuple[str, ...] | None:
        self.validate(fields)
        selected = None if selected_order is None else _normalize_order(selected_order)
        if self.mode is OrderMode.FIXED:
            if selected is not None and selected != self.orders[0]:
                raise OrderSelectionError('selected order differs from the fixed order')
            return self.orders[0]
        if self.mode is OrderMode.PRESELECTED:
            if selected is None:
                raise OrderSelectionError('preselected policy requires selected_order before decoding')
            if selected not in self.orders:
                raise OrderSelectionError('selected order is not declared by the policy')
            return selected
        if selected is not None:
            raise OrderSelectionError('dynamic policy does not accept a preselected order')
        return None

    def allowed_next(self, fields: Iterable[str], emitted: tuple[str, ...], selected_order: tuple[str, ...] | None) -> tuple[str, ...]:
        names = tuple(fields)
        if len(set(emitted)) != len(emitted) or any((name not in names for name in emitted)):
            raise OrderSelectionError('emitted field history is invalid')
        if self.mode in {OrderMode.FIXED, OrderMode.PRESELECTED}:
            order = self.resolve_selection(names, selected_order)
            assert order is not None
            if emitted != order[:len(emitted)]:
                raise OrderSelectionError('emitted fields are not a prefix of selected order')
            return () if len(emitted) == len(order) else (order[len(emitted)],)
        if not self.orders:
            used = set(emitted)
            return tuple((name for name in names if name not in used))
        candidates = {order[len(emitted)] for order in self.orders if len(emitted) < len(order) and order[:len(emitted)] == emitted}
        return tuple((name for name in names if name in candidates))

    def to_payload(self) -> dict[str, object]:
        return {'mode': self.mode.value, 'orders': [list(order) for order in self.orders]}

    @classmethod
    def from_payload(cls, payload: object) -> 'OrderPolicy':
        if not isinstance(payload, dict):
            raise OrderPolicyError('order policy payload must be an object')
        try:
            mode = OrderMode(payload['mode'])
            raw_orders = payload['orders']
            if not isinstance(raw_orders, list):
                raise TypeError('orders must be a list')
            if any((not isinstance(order, list) or any((type(name) is not str for name in order)) for order in raw_orders)):
                raise TypeError('every order must be a list of strings')
            orders = tuple((tuple(order) for order in raw_orders))
        except (KeyError, TypeError, ValueError) as error:
            raise OrderPolicyError('invalid order policy payload') from error
        return cls(mode, orders)

    @property
    def fingerprint(self) -> str:
        encoded = json.dumps(self.to_payload(), sort_keys=True, separators=(',', ':')).encode('ascii')
        return hashlib.sha256(encoded).hexdigest()
