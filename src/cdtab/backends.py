from dataclasses import dataclass
from types import MappingProxyType
import hashlib
import json
from cdtab.primitives import *
from cdtab.errors import ConstraintTypeError, EmptyRelationError, UnsupportedConstraintError
from cdtab.order import OrderPolicy
from cdtab.resources import ResourceBudget
from cdtab.compiler import CompileLimits, ResidualProgram, admit_spec
from cdtab.compiler import compile_constraints as legacy_compile
from cdtab.projection import Factor, DifferenceFactor, TableFactor, linear_atom, difference_edges
from cdtab.domains import IntSet
from cdtab.frontend import ExtendedSpec, Category, Nullable, IsValue, IsMissing, Or, Not, Implies, IfElse, COMPARISONS, base_type, plain_domain, field_domain, null_code, encode_value

@dataclass(frozen=True)
class ExtensionLimits(CompileLimits):
    max_branches: int = 32
    max_interval_fragments: int = 4096
    max_category_bytes: int = 100000
    max_codec_nodes: int = 4096
    max_prompt_bytes: int = 8192

    def __post_init__(self):
        super().__post_init__()
        for name in ('max_branches', 'max_interval_fragments', 'max_category_bytes', 'max_codec_nodes', 'max_prompt_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError('positive extension limits required')

def bounded_set(intervals, limit):
    if len(intervals) > limit:
        raise UnsupportedConstraintError('interval fragmentation budget exceeded')
    result = IntSet(tuple(intervals))
    if len(result.intervals) > limit:
        raise UnsupportedConstraintError('interval fragmentation budget exceeded')
    return result

def negate_atom(atom):
    cls = {Le: Gt, Lt: Ge, Ge: Lt, Gt: Le}.get(type(atom))
    if cls is not None:
        return ((cls(atom.left, atom.right),),)
    return ((Lt(atom.left, atom.right),), (Gt(atom.left, atom.right),))

def branch_product(left, right, limits):
    if len(left) * len(right) > limits.max_branches:
        raise UnsupportedConstraintError('Boolean alternative budget exceeded')
    return tuple((a + b for a in left for b in right))

def branch_union(parts, limits):
    result = tuple((branch for part in parts for branch in part))
    if len(result) > limits.max_branches:
        raise UnsupportedConstraintError('Boolean alternative budget exceeded')
    return result

def lower(predicate, types, limits, budget, negated=False, depth=0, count=None):
    budget.check()
    count = [0] if count is None else count
    count[0] += 1
    if count[0] > limits.max_ast_nodes or depth > limits.max_ast_depth:
        raise UnsupportedConstraintError('extended AST node/depth budget exceeded')

    def recur(p, neg=False):
        return lower(p, types, limits, budget, neg, depth + 1, count)
    kind = type(predicate)
    if kind is Not:
        return recur(predicate.predicate, not negated)
    if kind is Implies:
        return recur(Or((Not(predicate.condition), predicate.consequence)), negated)
    if kind is IfElse:
        return recur(Or((And((predicate.condition, predicate.when_true)), And((Not(predicate.condition), predicate.when_false)))), negated)
    if kind in (And, Or):
        if type(predicate.predicates) is not tuple or not predicate.predicates:
            raise ConstraintTypeError('immutable nonempty Boolean operands required')
        union = (kind is Or) != negated
        parts = [recur(p, negated) for p in predicate.predicates]
        if union:
            return branch_union(parts, limits)
        result = ((),)
        for part in parts:
            result = branch_product(result, part, limits)
        return result
    if kind in (IsValue, IsMissing):
        if predicate.field not in types:
            raise ConstraintTypeError('undeclared categorical/null field')
        field_kind = types[predicate.field]
        code = null_code(field_kind) if kind is IsMissing else encode_value(field_kind, predicate.value)
        atom = Eq(Var(predicate.field), Const(code, field_kind.scale))
        return negate_atom(atom) if negated else ((atom,),)
    if kind not in COMPARISONS:
        raise ConstraintTypeError('unregistered predicate')
    if not predicate.variables() <= set(types):
        raise ConstraintTypeError('undeclared numeric variable')
    if any((type(base_type(types[n])) is Category for n in predicate.variables())):
        raise ConstraintTypeError('categorical arithmetic is forbidden')
    predicate.validate(types)
    nullable = [n for n in sorted(predicate.variables()) if type(types[n]) is Nullable]
    guards = tuple((Ge(Var(n), Const(plain_domain(types[n]).minimum, types[n].scale)) for n in nullable))
    if not negated:
        return ((predicate,) + guards,)
    numeric = tuple((branch + guards for branch in negate_atom(predicate)))
    missing = tuple(((Eq(Var(n), Const(null_code(types[n]), types[n].scale)),) for n in nullable))
    return branch_union((numeric, missing), limits)

def conjunction_parts(predicate):
    if type(predicate) is And:
        return tuple((p for child in predicate.predicates for p in conjunction_parts(child)))
    return (predicate,)

def preflight(predicate, types, limits, budget):
    nodes = [0]
    expressions = (Var, Const, Add, Sub, MulBounded, MulConst, RoundHalfEvenDiv)

    def visit(node, depth=0):
        budget.check()
        nodes[0] += 1
        if nodes[0] > limits.max_ast_nodes or depth > limits.max_ast_depth:
            raise UnsupportedConstraintError('whole extended AST node/depth budget exceeded')
        kind = type(node)
        if kind in (And, Or):
            if type(node.predicates) is not tuple or not node.predicates:
                raise ConstraintTypeError('Boolean operands must be immutable and nonempty')
            children = node.predicates
        elif kind is Not:
            children = (node.predicate,)
        elif kind is Implies:
            children = (node.condition, node.consequence)
        elif kind is IfElse:
            children = (node.condition, node.when_true, node.when_false)
        elif kind in COMPARISONS:
            children = (node.left, node.right)
        elif kind in (Add, Sub, MulBounded):
            children = (node.left, node.right)
        elif kind in (MulConst, RoundHalfEvenDiv):
            children = (node.expr,)
        elif kind in (IsValue, IsMissing):
            if type(node.field) is not str or node.field not in types:
                raise ConstraintTypeError('undeclared typed field')
            null_code(types[node.field]) if kind is IsMissing else encode_value(types[node.field], node.value)
            children = ()
        elif kind is Var:
            if type(node.name) is not str or node.name not in types:
                raise ConstraintTypeError('undeclared numerical field')
            children = ()
        elif kind is Const:
            children = ()
        else:
            raise ConstraintTypeError('unregistered extended AST object')
        for child in children:
            visit(child, depth + 1)
    visit(predicate)
    return nodes[0]

def numeric_schema(schema):
    fields = []
    for field in schema.fields:
        kind, domain = (field.field_type, field_domain(field.field_type))
        base = base_type(kind)
        if type(kind) is Nullable or type(base) is Category:
            if type(base) is FiniteEnum or type(base) is Category:
                internal = FiniteEnum(domain.enumerate(limit=100000), kind.scale)
            else:
                internal = FixedPoint(domain.minimum, domain.maximum, kind.scale)
        else:
            internal = kind
        fields.append(Field(field.name, internal))
    return Schema(tuple(fields))

def restrict_unary(atom, domain):
    if not domain:
        return IntSet()
    coefficients, constant = linear_atom(atom)
    if len(coefficients) != 1:
        return None
    coefficient = next(iter(coefficients.values()))
    if type(atom) is Eq:
        q, r = divmod(-constant, coefficient)
        return domain.intersect(IntSet.span(q, q)) if r == 0 else IntSet()
    sign = 1 if type(atom) in (Le, Lt) else -1
    a, rhs = (sign * coefficient, -sign * constant - int(type(atom) in (Lt, Gt)))
    if a > 0:
        return domain.intersect(IntSet.span(domain.minimum, rhs // a))
    return domain.intersect(IntSet.span(-(-rhs // a), domain.maximum))

def remove_implied(atoms, domains):
    equations = []
    for atom in atoms:
        if type(atom) is Eq:
            try:
                equations.append(linear_atom(atom))
            except UnsupportedConstraintError:
                pass
    kept, removed = ([], 0)
    for atom in atoms:
        implied = False
        if type(atom) is not Eq:
            try:
                coefs, constant = linear_atom(atom)
                sign = 1 if type(atom) in (Le, Lt) else -1
                coefs = {n: sign * c for n, c in coefs.items()}
                bound = -sign * constant - int(type(atom) in (Lt, Gt))
                for eq, ec in equations:
                    for ratio in (-1, 1):
                        if not coefs or any((eq.get(n, 0) != ratio * c for n, c in coefs.items())):
                            continue
                        rest = {n: c for n, c in eq.items() if n not in coefs}
                        if len(rest) > 1:
                            continue
                        lo = hi = -ec
                        if rest:
                            n, c = next(iter(rest.items()))
                            lo, hi = sorted((-ec - c * domains[n].minimum, -ec - c * domains[n].maximum))
                        maximum = max(lo // ratio, hi // ratio)
                        if maximum <= bound:
                            implied = True
            except UnsupportedConstraintError:
                pass
        if implied:
            removed += 1
        else:
            kept.append(atom)
    return (tuple(kept), removed)

class AtomKernel:

    def __init__(self, names, domains, atom, limits, budget):
        self.names, self.domains, self.atom, self.limits = (names, domains, atom, limits)
        try:
            self.coefficients, self.constant = linear_atom(atom)
        except UnsupportedConstraintError:
            self.coefficients = None
        self.table = None
        if self.coefficients is None or (type(atom) is Eq and any((abs(c) != 1 for c in self.coefficients.values()))):
            self.table = TableFactor(names, domains, (atom,), limits, budget)
        self.kind = 'local-table' if self.table is not None else 'exact-linear-atom'

    def project(self, domains):
        return {n: self.project_field(domains, n) for n in self.names}

    def project_field(self, domains, name):
        if any((not d for d in domains.values())):
            return IntSet()
        if self.table is not None:
            return self.table._project(domains)[name]
        coefficients, constant = (self.coefficients, self.constant)
        if not coefficients:
            valid = self.atom.evaluate({n: 0 for n in self.names})
            return domains[name] if valid else IntSet()
        coefficient = coefficients.get(name, 0)
        if coefficient == 0:
            return domains[name]
        if type(self.atom) is not Eq:
            sign = 1 if type(self.atom) in (Le, Lt) else -1
            minimum = sum((min(sign * c * domains[n].minimum, sign * c * domains[n].maximum) for n, c in coefficients.items() if n != name))
            rhs = -sign * constant - int(type(self.atom) in (Lt, Gt)) - minimum
            a = sign * coefficient
            bound = IntSet.span(domains[name].minimum, rhs // a) if a > 0 else IntSet.span(-(-rhs // a), domains[name].maximum)
            return domains[name].intersect(bound)
        sums = IntSet.span(0, 0)
        for n, c in coefficients.items():
            if n == name:
                continue
            terms = domains[n].intervals if c == 1 else tuple(((-b, -a) for a, b in reversed(domains[n].intervals)))
            if len(sums.intervals) * len(terms) > self.limits.max_interval_fragments:
                raise UnsupportedConstraintError('exact sum fragmentation budget exceeded')
            sums = bounded_set([(a + x, b + y) for a, b in sums.intervals for x, y in terms], self.limits.max_interval_fragments)
        possible = tuple(sorted(((min((-constant - a) // coefficient, (-constant - b) // coefficient), max((-constant - a) // coefficient, (-constant - b) // coefficient)) for a, b in sums.intervals)))
        return domains[name].intersect(IntSet(possible))

def forest_graph(names, atoms):
    adjacency = {('v', n): [] for n in names}
    parent = {}

    def root(n):
        parent.setdefault(n, n)
        while parent[n] != n:
            n = parent[n]
        return n
    for i, atom in enumerate(atoms):
        f = ('f', i)
        adjacency[f] = []
        for name in sorted(atom.variables()):
            v = ('v', name)
            a, b = (root(f), root(v))
            if a == b:
                return None
            parent[a] = b
            adjacency[f].append(v)
            adjacency[v].append(f)
    return adjacency

class TreeFactor(Factor):
    kind = 'exact-acyclic-factor-messages'

    def __init__(self, names, domains, atoms, graph, limits, budget):
        super().__init__(names, domains, limits.projection_cache_entries)
        self.graph, self.limits = (graph, limits)
        self.kernels = tuple((AtomKernel(tuple((n for n in names if n in a.variables())), domains, a, limits, budget) for a in atoms))
        self.passes = self.message_count = 0

    def _project(self, domains):
        messages, output = ({}, {})

        def send(node, target):
            self.message_count += 1
            if node[0] == 'v':
                value = domains[node[1]]
                for neighbor in self.graph[node]:
                    if neighbor != target:
                        value = value.intersect(messages[neighbor, node])
                return value
            kernel = self.kernels[node[1]]
            restricted = {n: domains[n] for n in kernel.names}
            for neighbor in self.graph[node]:
                if neighbor != target:
                    restricted[neighbor[1]] = restricted[neighbor[1]].intersect(messages[neighbor, node])
            return kernel.project_field(restricted, target[1])
        visited = set()
        for root in self.graph:
            if root in visited:
                continue
            order, parent = ([root], {root: None})
            for node in order:
                visited.add(node)
                for neighbor in self.graph[node]:
                    if neighbor != parent[node]:
                        parent[neighbor] = node
                        order.append(neighbor)
            for node in reversed(order[1:]):
                messages[node, parent[node]] = send(node, parent[node])
            for node in order:
                for neighbor in self.graph[node]:
                    if parent.get(neighbor) == node:
                        messages[node, neighbor] = send(node, neighbor)
                if node[0] == 'v':
                    value = domains[node[1]]
                    for neighbor in self.graph[node]:
                        value = value.intersect(messages[neighbor, node])
                    output[node[1]] = value
        self.passes += 1
        return output if all(output.values()) else {n: IntSet() for n in self.names}

    def describe(self):
        return {**super().describe(), 'factor_nodes': len(self.kernels), 'graph_edges': sum(map(len, self.graph.values())) // 2, 'message_count': self.message_count, 'passes': self.passes, 'local_table_rows': sum((len(k.table.rows) for k in self.kernels if k.table is not None))}

class UnionFactor(Factor):
    kind = 'exact-residual-alternative-union'

    def __init__(self, names, domains, alternatives, limits):
        super().__init__(names, domains, limits.projection_cache_entries)
        self.alternatives, self.limits = (tuple(alternatives), limits)

    def _project(self, domains):
        intervals = {n: [] for n in self.names}
        for restricted, kernel in self.alternatives:
            current = {n: domains[n].intersect(restricted[n]) for n in self.names}
            if any((not d for d in current.values())):
                continue
            projected = current if kernel is None else kernel._project(current)
            if not all(projected.values()):
                continue
            for n in self.names:
                intervals[n].extend(projected[n].intervals)
        return {n: bounded_set(items, self.limits.max_interval_fragments) for n, items in intervals.items()}

    def describe(self):
        return {**super().describe(), 'alternatives': len(self.alternatives), 'backends': [k.describe() if k is not None else {'kind': 'specialized-domain-box'} for _, k in self.alternatives]}

class ExtendedProgram(ResidualProgram):

    def __init__(self, spec, internal_spec, policy, domains, factors, limits):
        super().__init__(internal_spec, policy, domains, factors, limits)
        self.extended_spec = spec
        self.program_fingerprint = hashlib.sha256((spec.fingerprint + json.dumps(policy.to_payload(), sort_keys=True) + ':v07-exact').encode()).hexdigest()

    def is_accepting(self, state):
        self._validate(state)
        return len(state.assignments) == len(self.spec.schema.names) and self.extended_spec.evaluate(dict(state.assignments))

def compile_constraints(spec, *, order_policy=None, limits=None, compact=False):
    if type(spec) is ConstraintSpec and (not compact):
        return legacy_compile(spec, order_policy=order_policy, limits=limits)
    if type(spec) is ConstraintSpec:
        spec = ExtendedSpec(spec.schema, spec.constraint, spec.order)
    if type(spec) is not ExtendedSpec:
        raise ConstraintTypeError('registered specification required')
    limits = ExtensionLimits() if limits is None else limits
    if type(limits) is not ExtensionLimits:
        raise ConstraintTypeError('extensions require explicit ExtensionLimits')
    budget = ResourceBudget(limits)
    policy = OrderPolicy.fixed(spec.order) if order_policy is None else order_policy
    budget.check()
    if len(spec.schema.names) > limits.max_fields:
        budget.fail('schema exceeds max_fields')
    internal_schema = numeric_schema(spec.schema)
    internal_spec = ConstraintSpec(internal_schema, Eq(Const(0), Const(0)), spec.order)
    with budget.phase('extended_schema_admission'):
        domains = admit_spec(internal_spec, limits, budget)
        domains = {f.name: field_domain(f.field_type) for f in spec.schema.fields}
        policy.validate(spec.schema.names)
        category_bytes = sum((sum((len(x.encode('ascii')) for x in base_type(f.field_type).labels)) for f in spec.schema.fields if type(base_type(f.field_type)) is Category))
        if category_bytes > limits.max_category_bytes:
            budget.fail('category lexical bytes exceed budget')
        ast_nodes = preflight(spec.constraint, spec.schema.types, limits, budget)
    parts = conjunction_parts(spec.constraint)
    parent = {n: n for n in spec.schema.names}

    def root(n):
        while parent[n] != n:
            n = parent[n]
        return n
    for p in parts:
        names = sorted(p.variables())
        if not set(names) <= set(parent):
            raise ConstraintTypeError('undeclared variable')
        for n in names[1:]:
            parent[root(n)] = root(names[0])
    groups, constant_parts = ({}, [])
    for p in parts:
        names = sorted(p.variables())
        if names:
            groups.setdefault(root(names[0]), []).append(p)
        else:
            constant_parts.append(p)
    factors, removed_count, total_rows, total_table_bytes = ([], 0, 0, 0)
    with budget.phase('extended_component_compile'):
        for p in constant_parts:
            from cdtab.frontend import evaluate
            constant_branches = lower(p, spec.schema.types, limits, budget)
            for branch in constant_branches:
                for atom in branch:
                    admit_spec(ConstraintSpec(internal_schema, atom, spec.order), limits, budget)
            if not evaluate(p, {}, spec.schema.types):
                raise EmptyRelationError('constant relation is false')
        for clauses in groups.values():
            names = tuple((n for n in spec.schema.names if any((n in p.variables() for p in clauses))))
            branches = lower(clauses[0] if len(clauses) == 1 else And(tuple(clauses)), spec.schema.types, limits, budget)
            alternatives = []
            for atoms in branches:
                restricted = {n: domains[n] for n in names}
                remaining, impossible = ([], False)
                for atom in atoms:
                    variables = atom.variables()
                    if not variables:
                        impossible |= not atom.evaluate({})
                        continue
                    checked = ConstraintSpec(internal_schema, atom, spec.order)
                    admit_spec(checked, limits, budget)
                    if len(variables) == 1:
                        n = next(iter(variables))
                        try:
                            projected = restrict_unary(atom, restricted[n])
                        except UnsupportedConstraintError:
                            projected = None
                        if projected is not None:
                            restricted[n] = projected
                            impossible |= not projected
                            continue
                    remaining.append(atom)
                if impossible:
                    continue
                remaining, removed = remove_implied(tuple(remaining), restricted)
                removed_count += removed
                if not remaining:
                    kernel = None
                else:
                    edges = difference_edges(remaining)
                    contiguous = all((len(d.intervals) == 1 for d in restricted.values()))
                    try:
                        if contiguous and edges is not None:
                            kernel = DifferenceFactor(names, restricted, edges, limits.projection_cache_entries, budget)
                        else:
                            graph = forest_graph(names, remaining)
                            if graph is not None:
                                kernel = TreeFactor(names, restricted, remaining, graph, limits, budget)
                            else:
                                kernel = TableFactor(names, restricted, remaining, limits, budget)
                    except EmptyRelationError:
                        continue
                if kernel is not None and (not all(kernel._project(restricted).values())):
                    continue
                total_rows += len(kernel.rows) if type(kernel) is TableFactor else sum((len(k.table.rows) for k in kernel.kernels if k.table is not None)) if type(kernel) is TreeFactor else 0
                total_table_bytes += kernel.storage_bytes if type(kernel) is TableFactor else sum((k.table.storage_bytes for k in kernel.kernels if k.table is not None)) if type(kernel) is TreeFactor else 0
                if total_rows > limits.max_local_rows:
                    budget.fail('aggregate extension table rows exceed budget')
                if total_table_bytes > limits.max_table_bytes:
                    budget.fail('aggregate extension table storage exceeds budget')
                alternatives.append((MappingProxyType(restricted), kernel))
            if not alternatives:
                raise EmptyRelationError('logical component has no satisfying alternative')
            factors.append(UnionFactor(names, domains, alternatives, limits))
    program = ExtendedProgram(spec, internal_spec, policy, domains, factors, limits)
    program.compile_diagnostics = {**budget.snapshot(), 'proved_redundant_atoms_removed': removed_count, 'total_local_rows': total_rows, 'total_local_table_bytes': total_table_bytes, 'extended_ast_nodes': ast_nodes}
    program.start(selected_order=policy.orders[0] if policy.mode.value == 'preselected' else None)
    from cdtab.artifact import encode_artifact
    program._artifact_bytes = encode_artifact(program)
    return program
