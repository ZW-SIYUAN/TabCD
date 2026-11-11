import itertools
from pathlib import Path
import unittest

import torch

from cdtab import (Add, And, BoundedInt, Const, ConstraintSpec, Eq, Field, FiniteEnum,
                   Le, MulConst, OrderPolicy, Schema, StaticTokenAdapter, Var,
                   apply_exact_mask, compile_constraints, make_masker)
from cdtab.demo import compile_demo, cvr, domains, load_config
from cdtab.errors import InvalidTransitionError
from cdtab.frontend import Category, ExtendedSpec, IsValue, Or
from cdtab.session import OrderFlexibleDecodingSession


class ProjectionTests(unittest.TestCase):
    def check_all_prefixes(self, spec, valid_rows):
        program = compile_constraints(spec, order_policy=OrderPolicy.dynamic())
        names = spec.schema.names
        for order in itertools.permutations(names):
            for complete in valid_rows:
                state = program.start()
                assigned = {}
                for name in order:
                    expected = {row[name] for row in valid_rows
                                if all(row[key] == value for key, value in assigned.items())}
                    actual = program.allowed_values(state, name)
                    self.assertEqual({value for lo, hi in actual.intervals for value in range(lo, hi + 1)}, expected)
                    state = program.advance(state, name, complete[name])
                    assigned[name] = complete[name]
        return program

    def test_box(self):
        schema = Schema((Field("x", BoundedInt(0, 2)), Field("y", BoundedInt(0, 1))))
        spec = ConstraintSpec(schema, Eq(Const(0), Const(0)), schema.names)
        self.check_all_prefixes(spec, [dict(x=x, y=y) for x in range(3) for y in range(2)])

    def test_difference_bounds(self):
        schema = Schema((Field("x", BoundedInt(0, 2)), Field("y", BoundedInt(0, 2))))
        spec = ConstraintSpec(schema, Le(Var("x"), Var("y")), schema.names)
        program = self.check_all_prefixes(spec, [dict(x=x, y=y) for x in range(3) for y in range(3) if x <= y])
        self.assertEqual(type(program.factors[0]).__name__, "DifferenceFactor")

    def test_unit_affine(self):
        schema = Schema(tuple(Field(name, BoundedInt(0, 2)) for name in ("x", "y", "z")))
        spec = ConstraintSpec(schema, Eq(Add(Var("x"), Var("y")), Var("z")), schema.names)
        program = self.check_all_prefixes(spec, [dict(x=x, y=y, z=z) for x, y, z in itertools.product(range(3), repeat=3) if x + y == z])
        self.assertEqual(type(program.factors[0]).__name__, "UnitAffineFactor")

    def test_local_table(self):
        schema = Schema((Field("x", BoundedInt(0, 4)), Field("y", BoundedInt(0, 2))))
        spec = ConstraintSpec(schema, Eq(Var("x"), MulConst(Var("y"), 2)), schema.names)
        program = self.check_all_prefixes(spec, [dict(x=2*y, y=y) for y in range(3)])
        self.assertEqual(type(program.factors[0]).__name__, "TableFactor")

    def test_category_disjunction(self):
        schema = Schema((Field("kind", Category(("a", "b"))), Field("x", BoundedInt(0, 2))))
        rule = Or((And((IsValue("kind", "a"), Eq(Var("x"), Const(1)))),
                   And((IsValue("kind", "b"), Eq(Var("x"), Const(2))))))
        self.check_all_prefixes(ExtendedSpec(schema, rule, schema.names), [dict(kind=0, x=1), dict(kind=1, x=2)])

    def test_projection_reuse(self):
        schema = Schema((Field("x", BoundedInt(0, 2)), Field("y", BoundedInt(0, 2))))
        spec = ConstraintSpec(schema, Le(Var("x"), Var("y")), schema.names)
        program = compile_constraints(spec, order_policy=OrderPolicy.dynamic())
        state = program.start()
        expected = program.allowed_values(state, "x")
        hits = program.factors[0].cache_hits
        self.assertEqual(program.allowed_values(state, "x"), expected)
        self.assertGreater(program.factors[0].cache_hits, hits)


class MaskTests(unittest.TestCase):
    def make_fixture(self):
        schema = Schema((Field("x", FiniteEnum((15, 16))), Field("y", FiniteEnum((15, 16)))))
        spec = ConstraintSpec(schema, Eq(Var("x"), Var("y")), schema.names)
        program = compile_constraints(spec, order_policy=OrderPolicy.dynamic())
        payloads = {value: bytes([value]) for value in range(128)}
        payloads.update({201: b"15", 202: b"16", 203: b", y is ", 204: b"15, y is 15"})
        adapter = StaticTokenAdapter(payloads, eos_token_id=200)
        return make_masker(program, adapter), payloads

    def test_whole_token_masks_against_independent_language(self):
        masker, payloads = self.make_fixture()
        words = {f"{first} is {value}, {second} is {value}".encode("ascii")
                 for first, second in [("x", "y"), ("y", "x")] for value in (15, 16)}
        prefixes = {word[:i] for word in words for i in range(len(word) + 1)}
        for prefix in sorted(prefixes):
            with self.subTest(prefix=prefix):
                expected = {token for token, payload in payloads.items()
                            if any(word.startswith(prefix + payload) for word in words)}
                if prefix in words:
                    expected.add(masker.eos_token_id)
                state = masker.engine.state_for_prefix(prefix)
                self.assertEqual(set(masker.legal_token_ids(state)), expected)

    def test_eos_only_after_complete_row(self):
        masker, _ = self.make_fixture()
        session = OrderFlexibleDecodingSession(masker)
        with self.assertRaises(InvalidTransitionError):
            session.consume(masker.eos_token_id)
        for token in b"x is 15, y is 15":
            session.consume(token)
        self.assertEqual(session.consume(masker.eos_token_id), {"x": 15, "y": 15})
        with self.assertRaises(InvalidTransitionError):
            session.consume(32)

    def test_mask_preserves_legal_scores(self):
        scores = torch.tensor([0.3, -2.1, 9.0, 0.0])
        masked = apply_exact_mask(scores, [0, 2])
        self.assertTrue(torch.equal(masked[[0, 2]], scores[[0, 2]]))
        self.assertTrue(torch.isneginf(masked[[1, 3]]).all())


class DemoTests(unittest.TestCase):
    def test_synthetic_data_rules_and_independent_cvr(self):
        import pandas as pd
        config = load_config(Path(__file__).resolve().parents[1] / "configs/rules/syn-affine.json")
        frame = pd.DataFrame([dict(x1=15, x2=5, x3=25), dict(x1=0, x2=0, x3=0)])
        declared = domains(frame, config)
        self.assertEqual(cvr(frame, declared, config["rules"]), 0)
        compile_demo(frame, config)
        frame.loc[0, "x3"] = 26
        self.assertEqual(cvr(frame, declared, config["rules"]), 0.5)
        frame["x1"] = frame["x1"].astype(object)
        frame.loc[1, "x1"] = "placeholder"
        self.assertEqual(cvr(frame, declared, config["rules"]), 1)


if __name__ == "__main__":
    unittest.main()
