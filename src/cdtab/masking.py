"""Reuse exact suffix languages to construct whole-token masks."""

from bisect import bisect_left
from collections import OrderedDict
from collections.abc import Mapping
from functools import lru_cache

from cdtab.serializer import SemanticState, SerializerState
from cdtab.tokenizer import OrderFlexibleTokenMasker


def normalize(cylinders):
    groups = {}
    for digits, dot, lo, hi in cylinders:
        if lo <= hi:
            groups.setdefault((digits, dot), []).append((lo, hi))
    result = []
    for (digits, dot), intervals in sorted(
        groups.items(), key=lambda x: (x[0][0], -1 if x[0][1] is None else x[0][1])
    ):
        merged = []
        for lo, hi in sorted(intervals):
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = merged[-1][0], max(hi, merged[-1][1])
            else:
                merged.append((lo, hi))
        result.extend((digits, dot, lo, hi) for lo, hi in merged)
    return tuple(result)


def numeric_language(codec, prefix, allowed):
    intervals = (
        codec.domain.intervals
        if allowed is codec.domain
        else codec.domain.intersect(allowed).intervals
    )
    negative = prefix.startswith(b"-")
    unsigned = prefix[1:] if negative else prefix
    whole, dot, fraction = unsigned.partition(b".")
    scale = codec.scale
    if (
        len(whole) > codec._whole_digits
        or (whole and (not whole.isdigit()))
        or (len(whole) > 1 and whole[0] == 48)
        or (
            dot
            and (
                not scale
                or not whole
                or len(fraction) > scale
                or (fraction and (not fraction.isdigit()))
            )
        )
    ):
        return (), ()
    count = len(whole) + len(fraction)
    known = int(whole + fraction) if count else 0
    first = max(1, len(whole))
    last = len(whole) if dot or whole == b"0" else codec._whole_digits

    def cylinders(is_negative):
        found = []
        for width in range(first, last + 1):
            remaining = width + scale - count
            if remaining < 0:
                continue
            unit = codec._powers[remaining]
            base = known * unit
            start, end = base, (known + 1) * unit - 1
            if width > 1:
                start = max(start, codec._powers[width - 1 + scale])
            point = width - len(whole) if scale and (not dot) else None
            for lower, upper in intervals:
                if is_negative:
                    lower, upper = max(1, -upper), -lower
                else:
                    lower = max(0, lower)
                a, b = max(start, lower), min(end, upper)
                if a <= b:
                    found.append((remaining, point, a - base, b - base))
        return normalize(found)

    positive = cylinders(negative)
    minus = cylinders(True) if not prefix else ()
    return positive, minus


@lru_cache(maxsize=128)
def power10(exponent):
    return 10**exponent


def consume(language, byte):
    current, minus = language
    if byte == 45:
        return (minus, ()) if minus else ((), ())
    updated = []
    for digits, dot, lo, hi in current:
        if dot == 0:
            if byte == 46:
                updated.append((digits, None, lo, hi))
        elif 48 <= byte <= 57 and digits:
            unit = power10(digits - 1)
            base = (byte - 48) * unit
            a, b = max(lo, base), min(hi, base + unit - 1)
            if a <= b:
                updated.append((digits - 1, None if dot is None else dot - 1, a - base, b - base))
    return tuple(updated), ()


def accepts(language):
    return any(digits == 0 and dot is None and (lo == 0) for digits, dot, lo, hi in language[0])


class LazyTransitions(Mapping):
    def __init__(self, masker, state, ids):
        self.masker, self.state, self.ids = masker, state, ids
        self._values = {}

    def __iter__(self):
        return iter(self.ids)

    def __len__(self):
        return len(self.ids)

    def __contains__(self, key):
        try:
            i = bisect_left(self.ids, key)
        except TypeError:
            hash(key)
            return False
        return i < len(self.ids) and self.ids[i] == key

    def __getitem__(self, key):
        if key not in self:
            raise KeyError(key)
        if key not in self._values:
            if key == self.masker.eos_token_id:
                if not self.masker.engine.can_end(self.state):
                    raise RuntimeError("INVALID_LAZY_EOS")
                successor = None
            else:
                successor = self.masker.selected_successor(
                    self.state, self.masker.adapter.payload(key)
                )
            self._values[key] = successor
        return self._values[key]


class TokenMasker(OrderFlexibleTokenMasker):
    """Reuse numeric suffix languages to expose exact whole-token candidates."""

    def __init__(self, engine, adapter):
        super().__init__(engine, adapter)
        self._numeric_plans = OrderedDict()
        self._header_plans = OrderedDict()
        self._languages = OrderedDict()
        self._vector_tables = OrderedDict()
        self._numeric_literals = []
        self._numeric_exits = []
        stack = [(self._root, b"")]
        while stack:
            node, payload = stack.pop()
            if payload:
                self._numeric_literals.extend((i, payload) for i in node.terminal_ids)
            if 44 in node.children:
                self._numeric_exits.append((node.children[44], payload + b","))
            for byte, child in node.children.items():
                if byte in b"-.0123456789":
                    stack.append((child, payload + bytes([byte])))
        self._whole_masks = OrderedDict()
        self._local_delimiters = all(32 not in child.children for child, _ in self._numeric_exits)

    def _cached(self, cache, key, build):
        if key in cache:
            cache.move_to_end(key)
            return cache[key], True
        value = build()
        cache[key] = value
        if len(cache) > self.engine.program._limits.state_cache_entries:
            cache.popitem(last=False)
        return value, False

    def _numeric_plan_trie(self, node, language):
        def build():
            ids = []
            exits = []
            stack = [(node, language, b"")]
            while stack:
                cursor, lang, payload = stack.pop()
                ids.extend(cursor.terminal_ids)
                if accepts(lang) and 44 in cursor.children:
                    exits.append((cursor.children[44], payload + b","))
                for byte, child in cursor.children.items():
                    if byte not in b"-.0123456789":
                        continue
                    nxt = consume(lang, byte)
                    if nxt[0] or nxt[1]:
                        stack.append((child, nxt, payload + bytes((byte,))))
            return tuple(ids), tuple(exits)

        result, _ = self._cached(self._numeric_plans, (id(node), language), build)
        return result

    def _vector_table(self, digits, point, negative):
        import numpy as np

        def build():
            ids = []
            lower = []
            upper = []
            for token, payload in self._numeric_literals:
                if payload.startswith(b"-") != negative:
                    continue
                text = payload[1:] if negative else payload
                if b"-" in text or text.count(b".") > 1:
                    continue
                if b"." in text:
                    if point is None or text.index(b".") != point:
                        continue
                    raw = text.replace(b".", b"")
                else:
                    raw = text
                    if point is not None and len(raw) > point:
                        continue
                if raw and (not raw.isdigit()) or len(raw) > digits:
                    continue
                unit = 10 ** (digits - len(raw))
                base = int(raw or b"0") * unit
                ids.append(token)
                lower.append(base)
                upper.append(base + unit - 1)
            arrays = tuple(np.asarray(x, dtype=np.int64) for x in (ids, lower, upper))
            for array in arrays:
                array.setflags(write=False)
            return arrays

        return self._cached(self._vector_tables, (digits, point, negative), build)[0]

    def _numeric_plan(self, node, language):
        if (
            node is not self._root
            or any(d > 18 for group in language for d, _, _, _ in group)
            or all(d < 6 for group in language for d, _, _, _ in group)
        ):
            return self._numeric_plan_trie(node, language)

        def build():
            ids = set()
            exits = []
            for negative, group in enumerate(language):
                for digits, point, lo, hi in group:
                    tokens, lower, upper = self._vector_table(digits, point, bool(negative))
                    ids.update(tokens[(lower <= hi) & (upper >= lo)].tolist())
            for child, payload in self._numeric_exits:
                lang = language
                for byte in payload[:-1]:
                    lang = consume(lang, byte)
                    if not lang[0] and (not lang[1]):
                        break
                if accepts(lang):
                    exits.append((child, payload))
            return tuple(sorted(ids)), tuple(exits)

        result, _ = self._cached(self._numeric_plans, (id(node), language), build)
        return result

    def _header_plan(self, node, candidates, prefix):
        ids = []
        ends = []
        for name, header in candidates:
            suffix = header[len(prefix) :]

            def build(suffix=suffix):
                cursor = node
                partial = list(cursor.terminal_ids)
                for byte in suffix:
                    cursor = cursor.children.get(byte)
                    if cursor is None:
                        return tuple(partial), None
                    partial.extend(cursor.terminal_ids)
                return tuple(partial), cursor

            result, _ = self._cached(self._header_plans, (id(node), suffix), build)
            partial, child = result
            ids.extend(partial)
            if child is not None and child.children:
                ends.append((child, name, suffix))
        return tuple(ids), tuple(ends)

    def _walk(self, node, state, ids):
        candidates, codec, allowed = self.engine._context(state)
        s = state.serializer
        if s.phase == "header":
            partial, ends = self._header_plan(node, candidates, s.header_prefix)
            ids.update(partial)
            for child, name, _ in ends:
                value = SemanticState(
                    SerializerState("value", b"", name, b"", s.committed), state.residual
                )
                self._walk(child, value, ids)
        else:
            language, _ = self._cached(
                self._languages,
                (codec.scale, codec.domain.intervals, s.value_prefix, allowed.intervals),
                lambda: numeric_language(codec, s.value_prefix, allowed),
            )
            partial, exits = self._numeric_plan(node, language)
            ids.update(partial)
            if len(s.committed) + 1 < len(self.engine.serializer.codecs):
                for child, payload in exits:
                    if 32 not in child.children:
                        ids.update(child.terminal_ids)
                    else:
                        next_state = self.engine.feed_bytes(state, payload)
                        self._walk(child, next_state, ids)

    def selected_successor(self, state, payload):
        if not payload:
            return state
        candidates, codec, allowed = self.engine._context(state)
        s = state.serializer
        if s.phase == "value" and all(b in b"-.0123456789" for b in payload):
            successor = SemanticState(
                SerializerState(
                    "value", b"", s.current_field, s.value_prefix + payload, s.committed
                ),
                state.residual,
            )
            self.engine._context(successor)
            return successor
        if s.phase == "header":
            prefix = s.header_prefix + payload
            for name, header in candidates:
                if prefix == header:
                    updated = SerializerState("value", b"", name, b"", s.committed)
                elif header.startswith(prefix):
                    updated = SerializerState("header", prefix, None, b"", s.committed)
                elif prefix.startswith(header) and all(
                    b in b"-.0123456789" for b in prefix[len(header) :]
                ):
                    updated = SerializerState(
                        "value", b"", name, prefix[len(header) :], s.committed
                    )
                else:
                    continue
                successor = SemanticState(updated, state.residual)
                self.engine._context(successor)
                return successor
        return self.engine.feed_bytes(state, payload)

    def legal_token_transitions(self, state):
        if type(state) is not SemanticState or type(state.serializer) is not SerializerState:
            self.engine._context(state)
        if state.serializer.phase == "value" and self._local_delimiters:
            _, codec, allowed = self.engine._context(state)
            s = state.serializer
            language, _ = self._cached(
                self._languages,
                (codec.scale, codec.domain.intervals, s.value_prefix, allowed.intervals),
                lambda: numeric_language(codec, s.value_prefix, allowed),
            )
            last = len(s.committed) + 1 == len(self.engine.serializer.codecs)
            eos = last and self.engine.can_end(state)

            def build():
                partial, exits = self._numeric_plan(self._root, language)
                tokens = set(partial)
                if not last:
                    for child, _ in exits:
                        tokens.update(child.terminal_ids)
                if eos:
                    tokens.add(self.eos_token_id)
                return tuple(sorted(tokens))

            ids, _ = self._cached(self._whole_masks, (language, last, eos), build)
            return LazyTransitions(self, state, ids)
        ids = set()
        self._walk(self._root, state, ids)
        s = state.serializer
        if (
            s.phase == "value"
            and len(s.committed) + 1 == len(self.engine.serializer.codecs)
            and self.engine.can_end(state)
        ):
            ids.add(self.eos_token_id)
        return LazyTransitions(self, state, tuple(sorted(ids)))
