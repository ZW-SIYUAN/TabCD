"""Incremental constraint state for native GReaT token sequences."""

import json
import re
from decimal import Decimal, localcontext

from cdtab.backends import ExtendedProgram
from cdtab.codecs import parse_native_value
from cdtab.errors import InvalidTransitionError, TokenizerContractError
from cdtab.frontend import decode_value
from cdtab.serializer import SemanticState, SerializerState
from cdtab.session import OrderFlexibleDecodingSession


def schema_for(engine):
    p = engine.program
    return p.extended_spec.schema if type(p) is ExtendedProgram else p.spec.schema


def state_for_native_prompt(engine, prompt):
    """Admit complete numeric or typed condition fields before continuation."""
    limit = getattr(engine.program._limits, "max_prompt_bytes", 8192)
    if type(prompt) is not bytes or len(prompt) > limit or (not prompt.isascii()):
        raise InvalidTransitionError("native prompt outside registered ASCII/length contract")
    if not prompt.endswith(b","):
        return engine.state_for_prefix(prompt)
    pieces = prompt[:-1].split(b", ")
    if len(pieces) == 1 and type(engine.program) is not ExtendedProgram:
        return single_anchor_state(engine, prompt)
    schema = schema_for(engine)
    residual = engine.program.start()
    for piece in pieces:
        matches = [
            (n, piece[len(n) + 4 :])
            for n in schema.names
            if piece.startswith((n + " is ").encode("ascii"))
        ]
        if len(matches) != 1:
            raise InvalidTransitionError(
                "condition prefix must have one complete declared header per field"
            )
        name, payload = matches[0]
        value = parse_native_value(
            schema.field(name).field_type, engine.serializer.codecs[name], payload
        )
        residual = engine.program.advance(residual, name, value)
    state = SemanticState(
        SerializerState("header", b",", None, b"", residual.assignments), residual
    )
    engine._context(state)
    return state


class NativeSessionProcessor:
    """Bind native prompt bytes to an incremental decoding session."""

    def __init__(self, masker, native, prompt_ids):
        self.masker, self.native = masker, native
        self.ids = tuple(int(x) for x in prompt_ids)
        payloads = [masker.adapter.payload(t) for t in self.ids]
        if any(p is None for p in payloads):
            raise TokenizerContractError("special token in native prompt")
        self.prompt = b"".join(payloads)
        if masker.adapter.encode_prefix(self.prompt) != self.ids:
            raise TokenizerContractError("native prompt tokenization differs")
        if native.decode(
            list(self.ids), clean_up_tokenization_spaces=False, skip_special_tokens=False
        ) != self.prompt.decode("ascii"):
            raise TokenizerContractError("native prompt decoding differs")
        self.session = OrderFlexibleDecodingSession(masker)
        self.session.state = state_for_native_prompt(masker.engine, self.prompt)

    def finish(self, token_ids):
        ids = tuple(int(x) for x in token_ids)
        if ids[: len(self.ids)] != self.ids:
            raise InvalidTransitionError("native generation rewrote fixed conditions")
        for token in ids[len(self.ids) :]:
            self.session.consume(token)
        self.ids = ids
        if not self.session.finished or ids[-1] != self.masker.eos_token_id:
            raise InvalidTransitionError("native generation did not finish with legal EOS")
        payload = b"".join(self.masker.adapter.payload(t) for t in ids[:-1])
        decoded = self.native.decode(
            list(ids[:-1]), clean_up_tokenization_spaces=False, skip_special_tokens=False
        )
        if payload.decode("ascii") != decoded or not payload.startswith(self.prompt):
            raise TokenizerContractError("native row bytes differ")
        schema = schema_for(self.masker.engine)
        row = dict(self.session.row)
        return {
            "row_typed": {
                name: decode_value(schema.field(name).field_type, value)
                for name, value in row.items()
            }
        }


def verify_tokenizer_bridge(native, adapter):
    from transformers import GPT2Tokenizer, GPT2TokenizerFast

    if type(native) not in (GPT2Tokenizer, GPT2TokenizerFast):
        raise TokenizerContractError("unregistered native tokenizer class")
    slow = adapter._tokenizer
    if native.get_vocab() != slow.get_vocab() or set(native.all_special_ids) != set(
        slow.all_special_ids
    ):
        raise TokenizerContractError("native and registered token vocabularies differ")
    if native.eos_token_id != adapter.eos_token_id:
        raise TokenizerContractError("EOS IDs differ")
    decoder = None
    if native.is_fast:
        decoder = json.loads(native.backend_tokenizer.decoder.__getstate__())
        if decoder.get("type") != "ByteLevel":
            raise TokenizerContractError("native decoder is not ByteLevel")


def single_anchor_state(engine, prompt):
    """Preserve the exact spelling and scale of a single numeric anchor."""
    if type(prompt) is not bytes or len(prompt) > 512:
        raise InvalidTransitionError("native prompt must be at most 512 bytes")
    try:
        text = prompt.decode("ascii")
    except UnicodeDecodeError as error:
        raise InvalidTransitionError("native prompt is not ASCII") from error
    if not text.endswith(","):
        return engine.state_for_prefix(prompt)
    matches = [
        (n, text[len(n) + 4 : -1])
        for n in engine.program.spec.schema.names
        if text.startswith(n + " is ")
    ]
    if len(matches) != 1:
        raise InvalidTransitionError("native anchor must contain exactly one known field")
    field, number = matches[0]
    if len(number) > 160 or re.fullmatch("-?(?:0|[1-9][0-9]*)(?:\\.[0-9]+)?", number) is None:
        raise InvalidTransitionError("unsupported native numeric anchor spelling")
    scale = engine.program.spec.schema.field(field).field_type.scale
    with localcontext() as context:
        context.prec = 192
        scaled = Decimal(number) * 10**scale
        if scaled != scaled.to_integral_value():
            raise InvalidTransitionError("native anchor is not on the declared precision grid")
        value = int(scaled)
    return engine.state_for_anchor(field, value)
