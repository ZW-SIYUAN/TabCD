"""Data-derived schemas and an independent row-level constraint check."""
from decimal import Decimal, InvalidOperation
import json
from pathlib import Path
from . import Add, And, BoundedInt, Category, Const, Eq, ExtendedSpec, Field, Le, MulConst, OrderPolicy, Schema, Var, compile_constraints

def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))

def domains(frame, config):
    bounds = config['bounds']
    result = {}
    for name in frame.columns:
        if frame[name].dtype.kind in 'iu':
            pair = [int(frame[name].min()), int(frame[name].max())] if bounds == 'data' else bounds[name]
            result[name] = ('integer', tuple(pair))
        else:
            result[name] = ('category', tuple(sorted(frame[name].unique())))
    return result

def compile_demo(frame, config):
    fields = tuple((Field(name, BoundedInt(*domain) if kind == 'integer' else Category(domain)) for name, (kind, domain) in domains(frame, config).items()))
    rules = []
    for rule in config['rules']:
        if rule['op'] == 'le':
            rules.append(Le(Var(rule['left']), Var(rule['right'])))
        elif rule['op'] == 'affine':
            terms = [MulConst(Var(name), coefficient) for name, coefficient in rule['terms'].items()]
            total = Const(0)
            for term in terms:
                total = Add(total, term)
            rules.append(Eq(Var(rule['result']), total))
        else:
            raise ValueError('Unknown rule operator')
    spec = ExtendedSpec(Schema(fields), And(tuple(rules)), tuple(frame.columns))
    return compile_constraints(spec, order_policy=OrderPolicy.dynamic())

def valid_row(row, declared, rules):
    """Missing, unparseable, out-of-domain and rule-violating rows all fail."""
    values = {}
    for name, (kind, domain) in declared.items():
        value = row[name]
        if kind == 'category':
            if not isinstance(value, str) or value not in domain:
                return False
            values[name] = value
        else:
            try:
                value = Decimal(str(value))
                if not value.is_finite() or value != value.to_integral_value():
                    return False
                if not domain[0] <= value <= domain[1]:
                    return False
                values[name] = int(value)
            except (InvalidOperation, ValueError):
                return False
    for rule in rules:
        if rule['op'] == 'le':
            if values[rule['left']] > values[rule['right']]:
                return False
        elif rule['op'] == 'affine':
            if values[rule['result']] != sum((values[name] * coefficient for name, coefficient in rule['terms'].items())):
                return False
        else:
            raise ValueError('Unknown rule operator')
    return True

def cvr(frame, declared, rules):
    if frame.empty:
        raise ValueError('CVR requires at least one requested record')
    return sum((not valid_row(row, declared, rules) for row in frame.to_dict('records'))) / len(frame)
