from decimal import Decimal, localcontext
import re
from cdtab.errors import ConstraintTypeError, UnsupportedConstraintError
from cdtab.domains import IntSet
from cdtab.numeric import NumericCodec
from cdtab.frontend import Category, Nullable, base_type, plain_domain, field_domain, null_code

class FiniteTextCodec:

    def __init__(self, values, limits):
        self.values = tuple(values)
        self._code_words = dict(self.values)
        self.domain = IntSet.points((code for code, _ in self.values))
        self.scale = 0
        self._exact, prefixes = ({}, {})
        for code, word in self.values:
            if word in self._exact:
                raise ConstraintTypeError('ambiguous native categorical spelling')
            self._exact[word] = code
            for length in range(len(word) + 1):
                prefixes.setdefault(word[:length], []).append(code)
                if len(prefixes) > limits.max_codec_nodes:
                    raise UnsupportedConstraintError('finite text prefix budget exceeded')
        self.prefixes = {p: IntSet.points(codes) for p, codes in prefixes.items()}
        self._prefix_entries = {p: tuple(((c, self._code_words[c]) for c in codes)) for p, codes in prefixes.items()}

    def feasible(self, prefix, allowed):
        return bool(self.prefixes.get(prefix, IntSet()).intersect(allowed))

    def exact(self, prefix):
        return self._exact.get(prefix)

    def next_bytes(self, prefix, allowed):
        return tuple(sorted({word[len(prefix)] for code, word in self._prefix_entries.get(prefix, ()) if allowed.contains(code) and word.startswith(prefix) and (len(word) > len(prefix))}))

    def encode(self, code):
        try:
            return self._code_words[code]
        except KeyError as error:
            raise ValueError('categorical value outside domain') from error

    def candidate_spellings(self, prefix, allowed):
        return tuple((word for code, word in self._prefix_entries.get(prefix, ()) if allowed.contains(code)))

class NullableNumericCodec:

    def __init__(self, kind, limits):
        self.numeric = NumericCodec(plain_domain(kind), kind.scale)
        self.null = null_code(kind)
        self.text = FiniteTextCodec(((self.null, b'None'),), limits)
        self.domain, self.scale = (field_domain(kind), kind.scale)

    def feasible(self, prefix, allowed):
        return self.numeric.feasible(prefix, allowed) or self.text.feasible(prefix, allowed)

    def exact(self, prefix):
        return self.null if prefix == b'None' else self.numeric.exact(prefix)

    def next_bytes(self, prefix, allowed):
        return tuple(sorted(set(self.numeric.next_bytes(prefix, allowed)) | set(self.text.next_bytes(prefix, allowed))))

    def encode(self, code):
        return b'None' if code == self.null else self.numeric.encode(code)

    def candidate_spellings(self, prefix, allowed):
        return self.text.candidate_spellings(prefix, allowed)

def make_codec(kind, limits):
    base = base_type(kind)
    if type(base) is Category:
        values = [(i, v.encode('ascii')) for i, v in enumerate(base.labels)]
        if type(kind) is Nullable:
            values.append((null_code(kind), b'None'))
        return FiniteTextCodec(tuple(values), limits)
    if type(kind) is Nullable:
        return NullableNumericCodec(kind, limits)
    return NumericCodec(plain_domain(kind), kind.scale)

def parse_native_value(kind, codec, payload):
    base = base_type(kind)
    if type(base) is Category or payload == b'None':
        code = codec.exact(payload)
        if code is None:
            raise ConstraintTypeError('unknown native category or undeclared NULL')
        return code
    if len(payload) > 160 or re.fullmatch(b'-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?', payload) is None:
        raise ConstraintTypeError('unsupported native exact numeric spelling')
    with localcontext() as context:
        context.prec = 192
        value = Decimal(payload.decode('ascii')) * 10 ** kind.scale
        if value != value.to_integral_value():
            raise ConstraintTypeError('native condition is off precision grid')
        code = int(value)
    if not plain_domain(kind).contains(code):
        raise ConstraintTypeError('native condition outside numeric domain')
    return code
