from cdtab.errors import InvalidTransitionError, TokenizerContractError
from cdtab.session import OrderFlexibleDecodingSession
from cdtab.great_helpers import state_for_native_prompt as old_prompt
from cdtab.serializer import SemanticState, SerializerState
from cdtab.backends import ExtendedProgram
from cdtab.frontend import decode_value
from cdtab.codecs import parse_native_value

def schema_for(engine):
    p = engine.program
    return p.extended_spec.schema if type(p) is ExtendedProgram else p.spec.schema

def state_for_native_prompt(engine, prompt):
    limit = getattr(engine.program._limits, 'max_prompt_bytes', 8192)
    if type(prompt) is not bytes or len(prompt) > limit or (not prompt.isascii()):
        raise InvalidTransitionError('native prompt outside registered ASCII/length contract')
    if not prompt.endswith(b','):
        return engine.state_for_prefix(prompt)
    pieces = prompt[:-1].split(b', ')
    if len(pieces) == 1 and type(engine.program) is not ExtendedProgram:
        return old_prompt(engine, prompt)
    schema = schema_for(engine)
    residual = engine.program.start()
    for piece in pieces:
        matches = [(n, piece[len(n) + 4:]) for n in schema.names if piece.startswith((n + ' is ').encode('ascii'))]
        if len(matches) != 1:
            raise InvalidTransitionError('condition prefix must have one complete declared header per field')
        name, payload = matches[0]
        value = parse_native_value(schema.field(name).field_type, engine.serializer.codecs[name], payload)
        residual = engine.program.advance(residual, name, value)
    state = SemanticState(SerializerState('header', b',', None, b'', residual.assignments), residual)
    engine._context(state)
    return state

class NativeSessionProcessor:

    def __init__(self, masker, native, prompt_ids):
        self.masker, self.native = (masker, native)
        self.ids = tuple((int(x) for x in prompt_ids))
        payloads = [masker.adapter.payload(t) for t in self.ids]
        if any((p is None for p in payloads)):
            raise TokenizerContractError('special token in native prompt')
        self.prompt = b''.join(payloads)
        if masker.adapter.encode_prefix(self.prompt) != self.ids:
            raise TokenizerContractError('native prompt tokenization differs')
        if native.decode(list(self.ids), clean_up_tokenization_spaces=False, skip_special_tokens=False) != self.prompt.decode('ascii'):
            raise TokenizerContractError('native prompt decoding differs')
        self.session = OrderFlexibleDecodingSession(masker)
        self.session.state = state_for_native_prompt(masker.engine, self.prompt)
        self.calls = 0

    def finish(self, token_ids):
        ids = tuple((int(x) for x in token_ids))
        if ids[:len(self.ids)] != self.ids:
            raise InvalidTransitionError('native generation rewrote fixed conditions')
        for token in ids[len(self.ids):]:
            self.session.consume(token)
        self.ids = ids
        if not self.session.finished or ids[-1] != self.masker.eos_token_id:
            raise InvalidTransitionError('native generation did not finish with legal EOS')
        payload = b''.join((self.masker.adapter.payload(t) for t in ids[:-1]))
        decoded = self.native.decode(list(ids[:-1]), clean_up_tokenization_spaces=False, skip_special_tokens=False)
        if payload.decode('ascii') != decoded or not payload.startswith(self.prompt):
            raise TokenizerContractError('native row bytes differ')
        schema = schema_for(self.masker.engine)
        row = dict(self.session.row)
        return {'row_scaled': row, 'row_typed': {n: decode_value(schema.field(n).field_type, v) for n, v in row.items()}, 'emitted_order': tuple(row), 'native_prompt': self.prompt.decode('ascii'), 'decoded': decoded, 'mask_calls': self.calls, 'prompt_contract': 'great-multiple-typed-anchors-v07', 'eos_complete': True}
