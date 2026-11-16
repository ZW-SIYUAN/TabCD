"""Canonical integer interval sets used by exact projections."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IntSet:
    intervals: tuple[tuple[int, int], ...] = ()

    def __post_init__(self):
        merged = []
        for pair in self.intervals:
            if len(pair) != 2 or any(type(v) is not int for v in pair):
                raise TypeError("interval bounds must be exact integers")
        for lo, hi in sorted(self.intervals):
            if lo > hi:
                continue
            if merged and lo <= merged[-1][1] + 1:
                merged[-1] = merged[-1][0], max(hi, merged[-1][1])
            else:
                merged.append((lo, hi))
        object.__setattr__(self, "intervals", tuple(merged))

    @classmethod
    def span(cls, lo, hi):
        return cls(((lo, hi),))

    @classmethod
    def points(cls, values):
        return cls(tuple((v, v) for v in values))

    def __bool__(self):
        return bool(self.intervals)

    @property
    def cardinality(self):
        return sum(hi - lo + 1 for lo, hi in self.intervals)

    @property
    def minimum(self):
        return self.intervals[0][0]

    @property
    def maximum(self):
        return self.intervals[-1][1]

    def contains(self, value):
        if type(value) is not int:
            return False
        lo, hi = 0, len(self.intervals)
        while lo < hi:
            mid = (lo + hi) // 2
            if self.intervals[mid][0] <= value:
                lo = mid + 1
            else:
                hi = mid
        return lo > 0 and value <= self.intervals[lo - 1][1]

    def intersect(self, other):
        result = []
        i = j = 0
        while i < len(self.intervals) and j < len(other.intervals):
            a, b = self.intervals[i]
            c, d = other.intervals[j]
            if max(a, c) <= min(b, d):
                result.append((max(a, c), min(b, d)))
            if b < d:
                i += 1
            else:
                j += 1
        return IntSet(tuple(result))

    def enumerate(self, *, limit):
        if self.cardinality > limit:
            raise ValueError("explicit enumeration exceeds declared limit")
        return tuple(v for lo, hi in self.intervals for v in range(lo, hi + 1))
