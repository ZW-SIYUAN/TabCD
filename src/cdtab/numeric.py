from functools import lru_cache
from cdtab.scaled import encode_scaled_integer, decode_scaled_integer
from cdtab.domains import IntSet

def _unsigned(prefix, lo, hi, scale):
    if lo > hi:
        return ()
    found = []
    max_whole = len(str(hi // 10 ** scale))
    for whole_digits in range(1, max_whole + 1):
        length = whole_digits + (1 + scale if scale else 0)
        if len(prefix) > length:
            continue
        digits = []
        valid = True
        for position, char in enumerate(prefix):
            if scale and position == whole_digits:
                if char != '.':
                    valid = False
                    break
            elif char not in '0123456789':
                valid = False
                break
            else:
                digits.append(char)
        if not valid:
            continue
        known = ''.join(digits)
        if whole_digits > 1 and known.startswith('0'):
            continue
        remaining = whole_digits + scale - len(known)
        base = int(known) if known else 0
        a, b = (base * 10 ** remaining, (base + 1) * 10 ** remaining - 1)
        if whole_digits > 1:
            a = max(a, 10 ** (whole_digits - 1 + scale))
        a, b = (max(lo, a), min(hi, b))
        if a <= b:
            found.append((a, b))
    return tuple(found)

@lru_cache(maxsize=8192)
def prefix_values(prefix: bytes, lo: int, hi: int, scale: int) -> IntSet:
    try:
        text = prefix.decode('ascii')
    except (AttributeError, UnicodeDecodeError):
        return IntSet()
    if lo > hi:
        return IntSet()
    result = []
    if text.startswith('-'):
        result.extend(((-b, -a) for a, b in _unsigned(text[1:], max(1, -hi), -lo, scale)))
    else:
        if text == '' and lo < 0:
            result.extend(((-b, -a) for a, b in _unsigned('', max(1, -hi), -lo, scale)))
        result.extend(_unsigned(text, max(0, lo), hi, scale))
    return IntSet(tuple(result))

class NumericCodec:

    def __init__(self, domain: IntSet, scale: int):
        self.domain, self.scale = (domain, scale)
        magnitude = max(abs(domain.minimum), abs(domain.maximum)) if domain else 0
        self._whole_digits = len(str(magnitude // 10 ** scale))
        self._powers = tuple((10 ** n for n in range(self._whole_digits + scale + 1)))
        self._allowed = None
        self._intervals = ()
        self._prefix = None
        self._answer = (False, ())

    def candidates(self, prefix):
        return self.domain.intersect(prefix_values(prefix, self.domain.minimum, self.domain.maximum, self.scale))

    def exact(self, prefix):
        try:
            value = decode_scaled_integer(prefix.decode('ascii'), self.scale)
        except (ValueError, UnicodeDecodeError):
            return None
        return value if self.domain.contains(value) else None

    def encode(self, value):
        if not self.domain.contains(value):
            raise ValueError('value outside numeric domain')
        return encode_scaled_integer(value, self.scale).encode('ascii')

    def next_bytes(self, prefix, allowed):
        return self._query(prefix, allowed)[1]

    def feasible(self, prefix, allowed):
        return self._query(prefix, allowed)[0]

    def _query(self, prefix, allowed):
        if allowed is not self._allowed:
            self._intervals = self.domain.intervals if allowed is self.domain else self.domain.intersect(allowed).intervals
            self._allowed = allowed
            self._prefix = None
        if prefix != self._prefix:
            self._answer = self._scan(prefix)
            self._prefix = prefix
        return self._answer

    def _scan(self, prefix):
        intervals = self._intervals
        if not intervals or not isinstance(prefix, bytes):
            return (False, ())
        negative = prefix.startswith(b'-')
        unsigned = prefix[1:] if negative else prefix
        whole, dot, fraction = unsigned.partition(b'.')
        scale = self.scale
        if len(whole) > self._whole_digits or (whole and (not whole.isdigit())) or (len(whole) > 1 and whole[0] == 48) or (dot and (not scale or not whole or len(fraction) > scale or (fraction and (not fraction.isdigit())))):
            return (False, ())
        count = len(whole) + len(fraction)
        known = int(whole + fraction) if count else 0
        first = max(1, len(whole))
        last = len(whole) if dot or whole == b'0' else self._whole_digits
        powers = self._powers
        bits = 1 if not prefix and intervals[0][0] < 0 else 0
        feasible = bool(bits)
        for width in range(first, last + 1):
            remaining = width + scale - count
            if remaining < 0:
                continue
            unit = powers[remaining]
            start, end = (known * unit, (known + 1) * unit - 1)
            if width > 1:
                start = max(start, powers[width - 1 + scale])
            decimal_next = bool(scale and (not dot) and (len(whole) == width))
            digit_next = remaining > 0 and (not decimal_next)
            place = powers[remaining - 1] if digit_next else 1
            for lower, upper in intervals:
                if negative:
                    lower, upper = (max(1, -upper), -lower)
                else:
                    lower = max(0, lower)
                a, b = (max(start, lower), min(end, upper))
                if a > b:
                    continue
                feasible = True
                if decimal_next:
                    bits |= 2
                elif digit_next:
                    low, high = (a // place % 10, b // place % 10)
                    bits |= (1 << high - low + 1) - 1 << low + 2
        return (feasible, tuple((byte for i, byte in enumerate(b'-.0123456789') if bits & 1 << i)))
