from collections import OrderedDict
from itertools import product
import math
import sys
from types import MappingProxyType
from cdtab.primitives import Add, Sub, MulConst, Var, Const, Eq, Le, Lt, Gt
from cdtab.errors import EmptyRelationError, UnsupportedConstraintError
from cdtab.domains import IntSet

def affine(expr):
    if type(expr) is Var:
        return ({expr.name: 1}, 0)
    if type(expr) is Const:
        return ({}, expr.value)
    if type(expr) is MulConst:
        coefficients, constant = affine(expr.expr)
        return ({n: v * expr.coefficient for n, v in coefficients.items() if v * expr.coefficient}, constant * expr.coefficient)
    if type(expr) in (Add, Sub):
        left, a = affine(expr.left)
        right, b = affine(expr.right)
        sign = 1 if type(expr) is Add else -1
        result = dict(left)
        for name, value in right.items():
            result[name] = result.get(name, 0) + sign * value
        return ({n: v for n, v in result.items() if v}, a + sign * b)
    raise UnsupportedConstraintError('expression is not affine')

def linear_atom(atom):
    return affine(Sub(atom.left, atom.right))

def difference_edges(atoms):
    edges = []
    for atom in atoms:
        try:
            coefficients, constant = linear_atom(atom)
        except UnsupportedConstraintError:
            return None
        signs = (1, -1) if type(atom) is Eq else (1,) if type(atom) in (Le, Lt) else (-1,)
        for sign in signs:
            c = {n: sign * v for n, v in coefficients.items()}
            if any((v not in (-1, 1) for v in c.values())):
                return None
            positive, negative = ([n for n, v in c.items() if v == 1], [n for n, v in c.items() if v == -1])
            if len(positive) > 1 or len(negative) > 1:
                return None
            bound = -sign * constant - (1 if type(atom) in (Lt, Gt) else 0)
            edges.append((positive[0] if positive else None, negative[0] if negative else None, bound))
    return edges

class Factor:
    kind = 'abstract'

    def __init__(self, names, domains, cache_entries):
        self.names = tuple(names)
        self.domains = MappingProxyType({n: domains[n] for n in names})
        self._cache = OrderedDict()
        self._cache_bytes = 0
        self.cache_entries = cache_entries
        self.queries = self.cache_hits = 0

    def project(self, assignments):
        key = tuple(((n, assignments[n]) for n in self.names if n in assignments))
        self.queries += 1
        if key in self._cache:
            self.cache_hits += 1
            self._cache.move_to_end(key)
            return self._cache[key][0]
        domains = dict(self.domains)
        for name, value in key:
            if not domains[name].contains(value):
                return {n: IntSet() for n in self.names}
            domains[name] = IntSet.span(value, value)
        result = MappingProxyType(self._project(domains))
        size = sys.getsizeof(key) + sum((sys.getsizeof(pair) for pair in key))
        size += sys.getsizeof(dict(result)) + sum((sys.getsizeof(v) + sys.getsizeof(v.intervals) + sum((sys.getsizeof(pair) + sum((sys.getsizeof(x) for x in pair)) for pair in v.intervals)) for v in result.values()))
        if size <= 1024 ** 2:
            self._cache[key] = (result, size)
            self._cache_bytes += size
        while len(self._cache) > self.cache_entries or self._cache_bytes > 1024 ** 2:
            _, (_, removed) = self._cache.popitem(last=False)
            self._cache_bytes -= removed
        return result

    def describe(self):
        return {'kind': self.kind, 'fields': self.names, 'stored_rows': 0}

def close_matrix(matrix, checkpoint=lambda: None):
    n = len(matrix)
    for k in range(n):
        checkpoint()
        for i in range(n):
            if matrix[i][k] is None:
                continue
            for j in range(n):
                if matrix[k][j] is not None:
                    value = matrix[i][k] + matrix[k][j]
                    if matrix[i][j] is None or value < matrix[i][j]:
                        matrix[i][j] = value
    return all((matrix[i][i] >= 0 for i in range(n)))

class DifferenceFactor(Factor):
    kind = 'integer-difference-bounds'

    def __init__(self, names, domains, edges, cache_entries, budget):
        super().__init__(names, domains, cache_entries)
        self.indices = {None: 0, **{n: i + 1 for i, n in enumerate(names)}}
        size = len(names) + 1
        matrix = [[0 if i == j else None for j in range(size)] for i in range(size)]
        for a, b, bound in edges:
            i, j = (self.indices[a], self.indices[b])
            matrix[i][j] = bound if matrix[i][j] is None else min(matrix[i][j], bound)
        for name, domain in self.domains.items():
            i = self.indices[name]
            matrix[i][0] = min(domain.maximum, matrix[i][0]) if matrix[i][0] is not None else domain.maximum
            matrix[0][i] = min(-domain.minimum, matrix[0][i]) if matrix[0][i] is not None else -domain.minimum
        if not close_matrix(matrix, budget.check):
            raise EmptyRelationError('inconsistent difference constraints')
        self.matrix = tuple((tuple(row) for row in matrix))

    def _project(self, domains):
        matrix = [list(row) for row in self.matrix]
        for name, domain in domains.items():
            i = self.indices[name]
            matrix[i][0] = min(matrix[i][0], domain.maximum)
            matrix[0][i] = min(matrix[0][i], -domain.minimum)
        if not close_matrix(matrix):
            return {n: IntSet() for n in self.names}
        return {n: IntSet.span(-matrix[0][self.indices[n]], matrix[self.indices[n]][0]) for n in self.names}

    def describe(self):
        return {**super().describe(), 'matrix_entries': (len(self.names) + 1) ** 2}

class UnitAffineFactor(Factor):
    kind = 'unit-affine-interval'

    def __init__(self, names, domains, coefficients, target, cache_entries):
        super().__init__(names, domains, cache_entries)
        self.coefficients, self.target = (coefficients, target)

    def _project(self, domains):
        terms = {n: (min(c * domains[n].minimum, c * domains[n].maximum), max(c * domains[n].minimum, c * domains[n].maximum)) for n, c in self.coefficients.items()}
        lower, upper = (sum((a for a, _ in terms.values())), sum((b for _, b in terms.values())))
        if not lower <= self.target <= upper:
            return {n: IntSet() for n in self.names}
        result = {}
        for name, domain in domains.items():
            coefficient = self.coefficients.get(name, 0)
            if coefficient == 0:
                result[name] = domain
                continue
            a, b = terms[name]
            other_low, other_high = (lower - a, upper - b)
            x, y = ((self.target - other_low) // coefficient, (self.target - other_high) // coefficient)
            result[name] = domain.intersect(IntSet.span(min(x, y), max(x, y)))
        return result

class TableFactor(Factor):
    kind = 'bounded-local-table'

    def __init__(self, names, domains, atoms, limits, budget):
        super().__init__(names, domains, limits.projection_cache_entries)
        output, expression = (None, None)
        for atom in atoms:
            if type(atom) is Eq:
                for left, right in ((atom.left, atom.right), (atom.right, atom.left)):
                    if type(left) is Var and left.name not in right.variables():
                        output, expression = (left.name, right)
                        break
            if output is not None:
                break
        inputs = tuple((n for n in names if n != output))
        trials = math.prod((domains[n].cardinality for n in inputs))
        if trials > limits.max_local_trials:
            budget.fail(f'unsupported large component {names}: needs {trials} local trials; no global fallback')
        if sum((domains[n].cardinality for n in inputs)) > limits.max_enum_values:
            budget.fail('local enumeration input domains exceed max_enum_values')
        values = [domains[n].enumerate(limit=limits.max_enum_values) for n in inputs]
        rows = []
        storage = 0
        for combination in product(*values):
            budget.check()
            env = dict(zip(inputs, combination))
            if output is not None:
                value = expression.evaluate(env)
                if not domains[output].contains(value):
                    continue
                env[output] = value
            if all((atom.evaluate(env) for atom in atoms)):
                if len(rows) >= limits.max_local_rows:
                    budget.fail('component exceeds max_local_rows')
                row = tuple((env[n] for n in names))
                storage += sys.getsizeof(row) + sum((sys.getsizeof(v) for v in row))
                if storage + sys.getsizeof(rows) > limits.max_table_bytes:
                    budget.fail('local table exceeds max_table_bytes')
                rows.append(row)
        if not rows:
            raise EmptyRelationError('local relation is empty')
        self.rows = tuple(rows)
        self.storage_bytes = storage + sys.getsizeof(self.rows)
        self.trials, self.functional_output = (trials, output)

    def _project(self, domains):
        values = [set() for _ in self.names]
        for row in self.rows:
            if all((domains[n].contains(v) for n, v in zip(self.names, row))):
                for bucket, value in zip(values, row):
                    bucket.add(value)
        return {n: IntSet.points(bucket) for n, bucket in zip(self.names, values)}

    def describe(self):
        return {**super().describe(), 'stored_rows': len(self.rows), 'local_trials': self.trials, 'storage_bytes_conservative': self.storage_bytes, 'functional_output': self.functional_output}

def build_factor(names, domains, atoms, limits, budget):
    contiguous = all((len(domains[n].intervals) == 1 for n in names))
    if contiguous:
        edges = difference_edges(atoms)
        if edges is not None:
            return DifferenceFactor(names, domains, edges, limits.projection_cache_entries, budget)
        if len(atoms) == 1 and type(atoms[0]) is Eq:
            try:
                coefficients, constant = linear_atom(atoms[0])
            except UnsupportedConstraintError:
                coefficients = None
            if coefficients is not None and all((c in (-1, 1) for c in coefficients.values())):
                return UnitAffineFactor(names, domains, coefficients, -constant, limits.projection_cache_entries)
    return TableFactor(names, domains, atoms, limits, budget)
