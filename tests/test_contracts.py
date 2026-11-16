"""Regression tests for input admission, resource limits, and exact masks."""

import itertools
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from cdtab import (
    And,
    BoundedInt,
    Category,
    CompileLimits,
    Const,
    ConstraintSpec,
    Eq,
    ExtendedSpec,
    Field,
    FiniteEnum,
    GPT2ByteBPEAdapter,
    IsValue,
    MulConst,
    Or,
    OrderPolicy,
    Schema,
    StaticTokenAdapter,
    Var,
    compile_constraints,
    make_masker,
)
from cdtab.backends import ExtensionLimits
from cdtab.errors import (
    ArtifactIntegrityError,
    ConstraintTypeError,
    InvalidTransitionError,
    RelationMaterializationLimitError,
    TokenizerContractError,
)
from cdtab.resources import ResourceBudget


class AdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from transformers import GPT2Tokenizer, GPT2TokenizerFast
        from transformers.models.gpt2.tokenization_gpt2 import bytes_to_unicode

        cls.directory = tempfile.TemporaryDirectory()
        directory = Path(cls.directory.name)
        vocab = {value: key for key, value in bytes_to_unicode().items()}
        vocab["<|endoftext|>"] = 256
        (directory / "vocab.json").write_text(json.dumps(vocab), encoding="utf-8")
        (directory / "merges.txt").write_text("#version: 0.2\n", encoding="utf-8")
        options = dict(
            vocab_file=str(directory / "vocab.json"), merges_file=str(directory / "merges.txt")
        )
        cls.slow = GPT2Tokenizer(**options)
        cls.fast = GPT2TokenizerFast(**options)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_slow_tokenizer_preserves_byte_payloads(self):
        adapter = GPT2ByteBPEAdapter(self.slow)
        text = b"x is 15, y is 15"
        tokens = adapter.encode_prefix(text)
        self.assertEqual(b"".join(adapter.payload(token) for token in tokens), text)

    def test_fast_adapter_is_rejected(self):
        with self.assertRaises(TokenizerContractError):
            GPT2ByteBPEAdapter(self.fast)

    def test_native_warmup_preserves_the_sampling_stream(self):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, set_seed

        from run import generate

        set_seed(2207)
        model = GPT2LMHeadModel(
            GPT2Config(
                vocab_size=257,
                n_positions=32,
                n_embd=16,
                n_layer=1,
                n_head=2,
                bos_token_id=256,
                eos_token_id=256,
                pad_token_id=256,
            )
        )
        original_generate = model.generate
        outputs = []

        def record(**kwargs):
            result = original_generate(**kwargs)
            outputs.append(result.clone())
            return result

        model.generate = record
        self.slow.pad_token = self.slow.eos_token
        great = SimpleNamespace(model=model, tokenizer=self.slow, columns=["x"])
        prompts = ["x is ", "x is "]
        threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            before, measured = generate(great, prompts, None, "cpu", 2, 16)
            _, warmup_time = generate(great, prompts[:1], None, "cpu", 2, 16, measure=False)
            after, _ = generate(great, prompts, None, "cpu", 2, 16)
        finally:
            torch.set_num_threads(threads)
        self.assertIsNone(warmup_time)
        self.assertGreaterEqual(measured, 0)
        self.assertTrue(torch.equal(outputs[0], outputs[2]))
        self.assertTrue(before.equals(after))

    def test_added_ordinary_tokens_are_rejected(self):
        import copy

        tokenizer = copy.deepcopy(self.slow)
        tokenizer.add_tokens(["new_field_token"])
        with self.assertRaises(TokenizerContractError):
            GPT2ByteBPEAdapter(tokenizer)

    def test_single_byte_coverage_is_required(self):
        schema = Schema((Field("x", BoundedInt(0, 2)),))
        spec = ConstraintSpec(schema, Eq(Const(0), Const(0)), schema.names)
        program = compile_constraints(spec)
        payloads = {token: bytes([token]) for token in range(128) if token != ord("1")}
        payloads[201] = b"11"
        with self.assertRaises(TokenizerContractError):
            make_masker(program, StaticTokenAdapter(payloads, eos_token_id=200))

    def test_category_grammar_rejections(self):
        for label in ("a,b", "a is b", "None", "placeholder", "", " a", "a\n"):
            with self.subTest(label=label), self.assertRaises(ConstraintTypeError):
                Category((label,))

    def test_local_trial_limit_is_enforced(self):
        schema = Schema((Field("x", BoundedInt(0, 6)), Field("y", BoundedInt(0, 3))))
        spec = ConstraintSpec(schema, Eq(Var("x"), MulConst(Var("y"), 2)), schema.names)
        with self.assertRaises(RelationMaterializationLimitError):
            compile_constraints(spec, limits=CompileLimits(max_local_trials=3))

    def test_serialized_size_limit_is_enforced(self):
        schema = Schema((Field("x", BoundedInt(0, 1)),))
        spec = ConstraintSpec(schema, Eq(Const(0), Const(0)), schema.names)
        with self.assertRaises(RelationMaterializationLimitError):
            compile_constraints(spec, limits=CompileLimits(max_artifact_bytes=1))

    def test_typed_size_limit_is_enforced(self):
        schema = Schema((Field("kind", Category(("a", "b"))),))
        spec = ExtendedSpec(schema, IsValue("kind", "a"), schema.names)
        with self.assertRaises(ArtifactIntegrityError):
            compile_constraints(spec, limits=ExtensionLimits(max_artifact_bytes=1))

    def test_state_from_another_program_is_rejected(self):
        schema = Schema((Field("x", BoundedInt(0, 2)),))
        first = compile_constraints(ConstraintSpec(schema, Eq(Var("x"), Const(1)), schema.names))
        second = compile_constraints(ConstraintSpec(schema, Eq(Var("x"), Const(2)), schema.names))
        with self.assertRaises(InvalidTransitionError):
            second.allowed_values(first.start(), "x")


class ResourceTests(unittest.TestCase):
    def test_rss_limit_is_enforced(self):
        process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=17))
        with patch("psutil.Process", return_value=process):
            with self.assertRaises(RelationMaterializationLimitError) as caught:
                ResourceBudget(CompileLimits(max_process_rss_bytes=16))
        self.assertEqual(caught.exception.resource_diagnostics["peak_observed_rss_bytes"], 17)

    def test_compile_deadline_is_enforced(self):
        with patch("cdtab.resources.time.perf_counter", return_value=0) as clock:
            budget = ResourceBudget(CompileLimits(max_compile_seconds=1))
            clock.return_value = 2
            with self.assertRaises(RelationMaterializationLimitError):
                budget.check(force=True)


class TypedMaskTests(unittest.TestCase):
    def test_nullable_masks_match_an_independent_finite_language(self):
        from cdtab.frontend import IsMissing, Nullable

        schema = Schema(
            (Field("kind", Category(("a", "b"))), Field("x", Nullable(BoundedInt(15, 16))))
        )
        rule = Or(
            (
                And((IsValue("kind", "a"), Eq(Var("x"), Const(15)))),
                And((IsValue("kind", "b"), IsMissing("x"))),
            )
        )
        program = compile_constraints(
            ExtendedSpec(schema, rule, schema.names), order_policy=OrderPolicy.dynamic()
        )
        payloads = {token: bytes([token]) for token in range(128)}
        payloads.update({201: b"None", 202: b"15", 203: b"a, x is 15"})
        masker = make_masker(program, StaticTokenAdapter(payloads, eos_token_id=200))
        words = set()
        for kind, value in (("a", "15"), ("b", "None")):
            row = {"kind": kind, "x": value}
            for order in itertools.permutations(row):
                words.add(", ".join(f"{name} is {row[name]}" for name in order).encode("ascii"))
        prefixes = {word[:length] for word in words for length in range(len(word) + 1)}
        for prefix in sorted(prefixes):
            expected = {
                token
                for token, payload in payloads.items()
                if any(word.startswith(prefix + payload) for word in words)
            }
            if prefix in words:
                expected.add(200)
            with self.subTest(prefix=prefix):
                state = masker.engine.state_for_prefix(prefix)
                self.assertEqual(set(masker.legal_token_ids(state)), expected)

    def test_typed_masks_match_an_independent_finite_language(self):
        schema = Schema((Field("kind", Category(("a", "b"))), Field("x", FiniteEnum((15, 16)))))
        rule = Or(
            (
                And((IsValue("kind", "a"), Eq(Var("x"), Const(15)))),
                And((IsValue("kind", "b"), Eq(Var("x"), Const(16)))),
            )
        )
        program = compile_constraints(
            ExtendedSpec(schema, rule, schema.names), order_policy=OrderPolicy.dynamic()
        )
        payloads = {token: bytes([token]) for token in range(128)}
        payloads.update({201: b"15", 202: b"16", 203: b"a, x is 15", 204: b", kind is "})
        masker = make_masker(program, StaticTokenAdapter(payloads, eos_token_id=200))
        words = set()
        for kind, value in (("a", 15), ("b", 16)):
            row = {"kind": kind, "x": value}
            for order in itertools.permutations(row):
                words.add(", ".join(f"{name} is {row[name]}" for name in order).encode("ascii"))
        prefixes = {word[:length] for word in words for length in range(len(word) + 1)}
        for prefix in sorted(prefixes):
            expected = {
                token
                for token, payload in payloads.items()
                if any(word.startswith(prefix + payload) for word in words)
            }
            if prefix in words:
                expected.add(200)
            with self.subTest(prefix=prefix):
                state = masker.engine.state_for_prefix(prefix)
                self.assertEqual(set(masker.legal_token_ids(state)), expected)


if __name__ == "__main__":
    unittest.main()
