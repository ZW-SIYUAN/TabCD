from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
import math
from types import MappingProxyType
from cdtab.primitives import Add, And, BoundedInt, Const, ConstraintSpec, Eq, Field, FiniteEnum, FixedPoint, Ge, Gt, Le, Lt, MulBounded, MulConst, RoundHalfEvenDiv, Schema, Sub, Var
from cdtab.ast import atomic_predicates, round_half_even_ratio
from cdtab.errors import ConstraintTypeError, EmptyRelationError, InvalidTransitionError, UnsupportedConstraintError
from cdtab.order import OrderPolicy
from cdtab.contracts import validate_field_headers
from cdtab.resources import ResourceBudget
from cdtab.domains import IntSet
from cdtab.projection import build_factor
COMPILER_CONTRACT = 'factorized-exact-v04.0'
SERIALIZER_CONTRACT = 'interval-canonical-ascii-v04.0'

@dataclass(frozen=True)
class CompileLimits:
    max_compile_seconds: float = 600.0
    max_process_rss_bytes: int = 2 * 1024 ** 3
    max_artifact_bytes: int = 256 * 1024 ** 2
    max_fields: int = 64
    rss_check_interval: int = 64
    max_numeric_digits: int = 64
    max_intermediate_digits: int = 128
    max_scale: int = 12
    max_enum_values: int = 100000
    max_local_trials: int = 2000000
    max_local_rows: int = 100000
    max_table_bytes: int = 256 * 1024 ** 2
    max_ast_nodes: int = 1024
    max_ast_depth: int = 32
    projection_cache_entries: int = 128
    state_cache_entries: int = 1024

    def __post_init__(self):
        if type(self.max_compile_seconds) not in (int, float) or not math.isfinite(self.max_compile_seconds) or self.max_compile_seconds <= 0:
            raise ValueError('max_compile_seconds must be finite and positive')
        for name in ('max_process_rss_bytes', 'max_artifact_bytes', 'max_fields', 'rss_check_interval', 'max_numeric_digits', 'max_intermediate_digits', 'max_scale', 'max_enum_values', 'max_local_trials', 'max_local_rows', 'max_table_bytes', 'max_ast_nodes', 'max_ast_depth', 'projection_cache_entries', 'state_cache_entries'):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f'{name} must be a positive integer')

def _integer(value, *, digits):
    if type(value) is not int or len(str(abs(value))) > digits:
        raise ConstraintTypeError('integer value is not exact or exceeds digit limit')

def admit_spec(spec, limits, budget):
    if type(spec) is not ConstraintSpec or type(spec.schema) is not Schema:
        raise ConstraintTypeError('only registered typed schemas are admitted')
    if type(spec.schema.fields) is not tuple or type(spec.order) is not tuple:
        raise ConstraintTypeError('schema/order containers must be immutable tuples')
    if len(spec.schema.names) > limits.max_fields:
        budget.fail('schema exceeds max_fields')
    validate_field_headers(spec)
    domains = {}
    enum_count = 0
    for field in spec.schema.fields:
        budget.check()
        if type(field) is not Field or type(field.name) is not str:
            raise ConstraintTypeError('invalid field')
        if len(field.name) > 256:
            budget.fail('field header exceeds 256 characters')
        kind = field.field_type
        if type(kind) not in (BoundedInt, FixedPoint, FiniteEnum):
            raise UnsupportedConstraintError('unregistered domain type')
        if type(kind.scale) is not int or not 0 <= kind.scale <= limits.max_scale:
            raise ConstraintTypeError('unsupported numeric scale')
        if type(kind) is FiniteEnum:
            if type(kind.members) is not tuple:
                raise ConstraintTypeError('enum members must be an immutable tuple')
            enum_count += len(kind.members)
            if enum_count > limits.max_enum_values:
                budget.fail('explicit enums exceed max_enum_values')
            for value in kind.members:
                _integer(value, digits=limits.max_numeric_digits)
            domain = IntSet.points(kind.members)
        else:
            lo, hi = (kind.minimum, kind.maximum) if type(kind) is BoundedInt else (kind.minimum_scaled, kind.maximum_scaled)
            _integer(lo, digits=limits.max_numeric_digits)
            _integer(hi, digits=limits.max_numeric_digits)
            domain = IntSet.span(lo, hi)
        if not domain:
            raise EmptyRelationError('empty field domain')
        domains[field.name] = domain
    nodes = 0

    def visit(node, depth=0):
        nonlocal nodes
        budget.check()
        nodes += 1
        if nodes > limits.max_ast_nodes or depth > limits.max_ast_depth:
            budget.fail('constraint AST exceeds node/depth limits')
        if type(node) is And:
            if type(node.predicates) is not tuple:
                raise ConstraintTypeError('predicate conjunction must be an immutable tuple')
            for child in node.predicates:
                visit(child, depth + 1)
            return
        if type(node) in (Eq, Le, Lt, Ge, Gt):
            visit(node.left, depth + 1)
            visit(node.right, depth + 1)
            return
        if type(node) is Var:
            domain = domains[node.name]
            return (domain.minimum, domain.maximum)
        if type(node) is Const:
            _integer(node.value, digits=limits.max_numeric_digits)
            if type(node.scale) is not int or not 0 <= node.scale <= limits.max_scale:
                raise ConstraintTypeError('unsupported constant scale')
            bounds = (node.value, node.value)
        elif type(node) in (Add, Sub, MulBounded):
            a, b = visit(node.left, depth + 1)
            c, d = visit(node.right, depth + 1)
            if type(node) is Add:
                bounds = (a + c, b + d)
            elif type(node) is Sub:
                bounds = (a - d, b - c)
            else:
                values = (a * c, a * d, b * c, b * d)
                bounds = (min(values), max(values))
        elif type(node) is MulConst:
            _integer(node.coefficient, digits=limits.max_numeric_digits)
            if type(node.coefficient_scale) is not int or not 0 <= node.coefficient_scale <= limits.max_scale:
                raise ConstraintTypeError('unsupported coefficient scale')
            a, b = visit(node.expr, depth + 1)
            bounds = (min(a * node.coefficient, b * node.coefficient), max(a * node.coefficient, b * node.coefficient))
        elif type(node) is RoundHalfEvenDiv:
            _integer(node.divisor, digits=limits.max_numeric_digits)
            if node.divisor <= 0 or type(node.output_scale) is not int or (not 0 <= node.output_scale <= limits.max_scale):
                raise ConstraintTypeError('invalid round-half-even contract')
            a, b = visit(node.expr, depth + 1)
            bounds = (round_half_even_ratio(a, node.divisor), round_half_even_ratio(b, node.divisor))
        else:
            raise UnsupportedConstraintError(f'unregistered AST node: {type(node).__name__}')
        for value in bounds:
            _integer(value, digits=limits.max_intermediate_digits)
        return bounds
    visit(spec.constraint)
    spec.constraint.validate(spec.schema.types)
    return domains

@dataclass(frozen=True)
class State:
    program_fingerprint: str
    assignments: tuple[tuple[str, int], ...] = ()
    selected_order: tuple[str, ...] | None = None

class ResidualProgram:

    def __init__(self, spec, policy, domains, factors, limits):
        self.spec, self.policy, self._limits = (spec, policy, limits)
        self.domains = MappingProxyType(dict(domains))
        self.factors = tuple(factors)
        self._factor_by_field = {n: factor for factor in factors for n in factor.names}
        identity = {'spec': spec.fingerprint, 'policy': policy.to_payload(), 'compiler': COMPILER_CONTRACT, 'serializer': SERIALIZER_CONTRACT}
        self.program_fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        self._validated = OrderedDict()
        self._identity_validated = OrderedDict()
        self.compile_diagnostics = {}

    def _validate(self, state):
        key = id(state)
        known = self._identity_validated.get(key)
        if known is not None and known[0] is state and (type(state) is State) and (state.program_fingerprint == self.program_fingerprint) and (known[1] is state.assignments) and (known[2] is state.selected_order):
            self._identity_validated.move_to_end(key)
            return
        self._validate_untrusted(state)
        self._identity_validated[key] = (state, state.assignments, state.selected_order)
        if len(self._identity_validated) > self._limits.state_cache_entries:
            self._identity_validated.popitem(last=False)

    def _validate_untrusted(self, state):
        if type(state) is not State or state.program_fingerprint != self.program_fingerprint:
            raise InvalidTransitionError('state belongs to a different compiled program')
        if type(state.assignments) is not tuple or any((type(item) is not tuple or len(item) != 2 or type(item[0]) is not str or (type(item[1]) is not int) for item in state.assignments)):
            raise InvalidTransitionError('malformed immutable assignment history')
        try:
            if state in self._validated:
                self._validated.move_to_end(state)
                return
        except TypeError as error:
            raise InvalidTransitionError('malformed immutable state') from error
        selection = self.policy.resolve_selection(self.spec.schema.names, state.selected_order)
        if selection != state.selected_order:
            raise InvalidTransitionError('state has invalid order selection')
        emitted, env = ((), {})
        for name, value in state.assignments:
            if name not in self.policy.allowed_next(self.spec.schema.names, emitted, selection):
                raise InvalidTransitionError('invalid field history')
            if not self.domains[name].contains(value):
                raise InvalidTransitionError('state value outside domain')
            env[name] = value
            emitted += (name,)
        if any((not all(factor.project(env).values()) for factor in self.factors)):
            raise InvalidTransitionError('state has no complete legal extension')
        self._validated[state] = True
        if len(self._validated) > self._limits.state_cache_entries:
            self._validated.popitem(last=False)

    def start(self, *, selected_order=None):
        selected = self.policy.resolve_selection(self.spec.schema.names, selected_order)
        state = State(self.program_fingerprint, (), selected)
        self._validate(state)
        return state

    def allowed_fields(self, state):
        self._validate(state)
        key = id(state)
        known = self._identity_validated[key]
        if len(known) == 4:
            return known[3]
        fields = self.policy.allowed_next(self.spec.schema.names, tuple((n for n, _ in state.assignments)), state.selected_order)
        self._identity_validated[key] = known + (fields,)
        return fields

    def allowed_values(self, state, field):
        if field not in self.allowed_fields(state):
            raise InvalidTransitionError('field is not available in this state')
        factor = self._factor_by_field.get(field)
        return self.domains[field] if factor is None else factor.project(dict(state.assignments))[field]

    def advance(self, state, field, value):
        if type(value) is not int or type(field) is not str:
            return self._advance_untrusted(state, field, value)
        if not self.allowed_values(state, field).contains(value):
            raise InvalidTransitionError('value has no satisfying completion')
        successor = State(self.program_fingerprint, state.assignments + ((field, value),), state.selected_order)
        self._identity_validated[id(successor)] = (successor, successor.assignments, successor.selected_order)
        if len(self._identity_validated) > self._limits.state_cache_entries:
            self._identity_validated.popitem(last=False)
        return successor

    def _advance_untrusted(self, state, field, value):
        if not self.allowed_values(state, field).contains(value):
            raise InvalidTransitionError('value has no satisfying completion')
        successor = State(self.program_fingerprint, state.assignments + ((field, value),), state.selected_order)
        self._validate(successor)
        return successor

    def is_accepting(self, state):
        self._validate(state)
        return len(state.assignments) == len(self.spec.schema.names) and self.spec.constraint.evaluate(dict(state.assignments))

    def diagnostics(self):
        return {'compiler': COMPILER_CONTRACT, 'program_fingerprint': self.program_fingerprint, 'fields': len(self.spec.schema.names), 'global_rows_materialized': 0, 'independent_fields': [n for n in self.spec.schema.names if n not in self._factor_by_field], 'components': [factor.describe() for factor in self.factors], 'compile': self.compile_diagnostics, 'runtime': {'projection_queries': sum((f.queries for f in self.factors)), 'projection_cache_hits': sum((f.cache_hits for f in self.factors)), 'cached_states': len(self._validated)}}

def compile_constraints(spec, *, order_policy=None, limits=None, _budget=None):
    limits = CompileLimits() if limits is None else limits
    policy = OrderPolicy.fixed(spec.order) if order_policy is None else order_policy
    budget = ResourceBudget(limits) if _budget is None else _budget
    with budget.phase('schema_admission'):
        domains = admit_spec(spec, limits, budget)
        policy.validate(spec.schema.names)
    with budget.phase('factor_partition'):
        atoms = atomic_predicates(spec.constraint)
        parent = {n: n for n in spec.schema.names}

        def root(n):
            while parent[n] != n:
                n = parent[n]
            return n
        for atom in atoms:
            variables = sorted(atom.variables())
            if not variables:
                if not atom.evaluate({}):
                    raise EmptyRelationError('constant predicate is false')
                continue
            for name in variables[1:]:
                parent[root(name)] = root(variables[0])
        groups = {}
        for atom in atoms:
            variables = sorted(atom.variables())
            if variables:
                groups.setdefault(root(variables[0]), []).append(atom)
    factors = []
    table_bytes = 0
    with budget.phase('component_compile'):
        for group in groups.values():
            variables = frozenset().union(*(a.variables() for a in group))
            names = tuple((n for n in spec.schema.names if n in variables))
            factor = build_factor(names, domains, tuple(group), limits, budget)
            if not all(factor.project({}).values()):
                raise EmptyRelationError('constraint component has no satisfying assignment')
            table_bytes += getattr(factor, 'storage_bytes', 0)
            if table_bytes > limits.max_table_bytes:
                budget.fail('total local table storage exceeds max_table_bytes')
            factors.append(factor)
    with budget.phase('finalize'):
        program = ResidualProgram(spec, policy, domains, factors, limits)
        program.start(selected_order=policy.orders[0] if policy.mode.value == 'preselected' else None)
        from cdtab.artifact_base import encode_artifact
        program._artifact_bytes = encode_artifact(program, budget)
    program.compile_diagnostics = budget.snapshot()
    return program
