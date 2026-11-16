"""Incremental byte parsing with a reusable semantic residual state."""

from collections import OrderedDict
from dataclasses import dataclass
from functools import lru_cache

from cdtab.compiler import State
from cdtab.errors import InvalidTransitionError
from cdtab.numeric import NumericCodec
from cdtab.resources import ResourceBudget


@dataclass(frozen=True)
class SerializerState:
    phase: str = "header"
    header_prefix: bytes = b""
    current_field: str | None = None
    value_prefix: bytes = b""
    committed: tuple[tuple[str, int], ...] = ()


@dataclass(frozen=True)
class SemanticState:
    serializer: SerializerState
    residual: State


class RowSerializer:
    def __init__(self, program):
        self.program = program
        self.spec = program.spec
        budget = ResourceBudget(program._limits)
        with budget.phase("interval_serializer_setup"):
            self.codecs = {
                n: NumericCodec(program.domains[n], program.spec.schema.field(n).field_type.scale)
                for n in program.spec.schema.names
            }
            required = set(b", -.0123456789")
            for name in self.codecs:
                budget.check()
                required.update(self.header(name, 0))
            self.required_bytes = frozenset(required)

    @staticmethod
    @lru_cache(maxsize=128)
    def _header(name, later):
        return ((", " if later else "") + name + " is ").encode("ascii")

    @staticmethod
    def header(name, depth):
        return RowSerializer._header(name, depth != 0)

    def serialize(self, row, order):
        order = tuple(order)
        if (
            len(order) != len(self.codecs)
            or set(order) != set(self.codecs)
            or set(row) != set(self.codecs)
        ):
            raise ValueError("row/order must include every field exactly once")
        if not self.spec.constraint.evaluate(row):
            raise ValueError("row violates declared relation")
        return b"".join(
            self.header(n, i) + self.codecs[n].encode(row[n]) for i, n in enumerate(order)
        )


class SemanticByteEngine:
    def __init__(self, program):
        self.program = program
        self.serializer = RowSerializer(program)
        self._legal_cache = OrderedDict()

    def start(self, *, selected_order=None):
        return SemanticState(SerializerState(), self.program.start(selected_order=selected_order))

    def _context(self, state):
        if type(state) is not SemanticState or type(state.serializer) is not SerializerState:
            raise InvalidTransitionError("unregistered semantic state")
        s = state.serializer
        self.program._validate(state.residual)
        if s.committed != state.residual.assignments:
            raise InvalidTransitionError("byte and semantic assignments diverged")
        if type(s.header_prefix) is not bytes or type(s.value_prefix) is not bytes:
            raise InvalidTransitionError("prefixes must be bytes")
        if s.phase == "header":
            if s.current_field is not None or s.value_prefix:
                raise InvalidTransitionError("malformed header state")
            candidates = tuple(
                (n, self.serializer.header(n, len(s.committed)))
                for n in self.program.allowed_fields(state.residual)
            )
            candidates = tuple((n, h) for n, h in candidates if h.startswith(s.header_prefix))
            if not candidates:
                raise InvalidTransitionError("header has no completion")
            return candidates, None, None
        if s.phase != "value" or s.header_prefix or s.current_field not in self.serializer.codecs:
            raise InvalidTransitionError("malformed value state")
        allowed = self.program.allowed_values(state.residual, s.current_field)
        codec = self.serializer.codecs[s.current_field]
        if not codec.feasible(s.value_prefix, allowed):
            raise InvalidTransitionError("numeric prefix has no legal completion")
        return (), codec, allowed

    def legal_next_bytes(self, state):
        if state in self._legal_cache:
            self._legal_cache.move_to_end(state)
            return self._legal_cache[state]
        candidates, codec, allowed = self._context(state)
        s = state.serializer
        if s.phase == "header":
            result = tuple(
                sorted(
                    {
                        header[len(s.header_prefix)]
                        for _, header in candidates
                        if len(header) > len(s.header_prefix)
                    }
                )
            )
        else:
            possible = set(codec.next_bytes(s.value_prefix, allowed))
            value = codec.exact(s.value_prefix)
            if (
                len(s.committed) + 1 < len(self.serializer.codecs)
                and value is not None
                and allowed.contains(value)
            ):
                possible.add(ord(","))
            result = tuple(sorted(possible))
        self._legal_cache[state] = result
        if len(self._legal_cache) > self.program._limits.state_cache_entries:
            self._legal_cache.popitem(last=False)
        return result

    def feed_byte(self, state, byte):
        if (
            type(byte) is not int
            or not 0 <= byte <= 255
            or byte not in self.legal_next_bytes(state)
        ):
            raise InvalidTransitionError("byte has no legal completion")
        s, residual = state.serializer, state.residual
        if s.phase == "header":
            prefix = s.header_prefix + bytes((byte,))
            completed = [
                n
                for n in self.program.allowed_fields(residual)
                if self.serializer.header(n, len(s.committed)) == prefix
            ]
            updated = (
                SerializerState("value", b"", completed[0], b"", s.committed)
                if completed
                else SerializerState("header", prefix, None, b"", s.committed)
            )
        elif byte == ord(","):
            value = self.serializer.codecs[s.current_field].exact(s.value_prefix)
            residual = self.program.advance(residual, s.current_field, value)
            updated = SerializerState("header", b",", None, b"", residual.assignments)
        else:
            updated = SerializerState(
                "value", b"", s.current_field, s.value_prefix + bytes((byte,)), s.committed
            )
        return SemanticState(updated, residual)

    def feed_bytes(self, state, payload):
        for byte in payload:
            state = self.feed_byte(state, byte)
        return state

    def state_for_prefix(self, payload, *, selected_order=None):
        return self.feed_bytes(self.start(selected_order=selected_order), payload)

    def state_for_anchor(self, field, value, *, selected_order=None):
        if len(self.serializer.codecs) <= 1:
            raise InvalidTransitionError("comma-terminated anchor requires another field")
        residual = self.program.advance(
            self.program.start(selected_order=selected_order), field, value
        )
        return SemanticState(
            SerializerState("header", b",", None, b"", residual.assignments), residual
        )

    def can_end(self, state):
        _, codec, allowed = self._context(state)
        s = state.serializer
        if s.phase != "value" or len(s.committed) + 1 != len(self.serializer.codecs):
            return False
        exact = codec.exact(s.value_prefix)
        return (
            exact is not None
            and allowed.contains(exact)
            and self.program.is_accepting(
                self.program.advance(state.residual, s.current_field, exact)
            )
        )

    def finish(self, state):
        if not self.can_end(state):
            raise InvalidTransitionError("row cannot end here")
        s = state.serializer
        return dict(
            s.committed
            + ((s.current_field, self.serializer.codecs[s.current_field].exact(s.value_prefix)),)
        )
