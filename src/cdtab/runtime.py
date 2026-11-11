from cdtab.numeric import NumericCodec
from cdtab.serializer import RowSerializer, SemanticByteEngine, SemanticState, SerializerState
from cdtab.masking import TokenMasker as LegacyMasker, LazyTransitions, numeric_language
from cdtab.backends import ExtendedProgram
from cdtab.codecs import make_codec, NullableNumericCodec, FiniteTextCodec
from cdtab.frontend import decode_value

class TypedSerializer(RowSerializer):

    def __init__(self, program):
        self.program, self.spec = (program, program.spec)
        self.codecs = {f.name: make_codec(f.field_type, program._limits) for f in program.extended_spec.schema.fields}
        required = set(b', -.0123456789')
        for name, codec in self.codecs.items():
            required.update(self.header(name, 0))
            text = codec.text if type(codec) is NullableNumericCodec else codec if type(codec) is FiniteTextCodec else None
            if text is not None:
                for _, word in text.values:
                    required.update(word)
        self.required_bytes = frozenset(required)
        self.setup_diagnostics = {'registered_codec_fields': len(self.codecs)}

    def serialize(self, row, order):
        if set(row) != set(self.program.extended_spec.schema.names):
            raise ValueError('typed row must include every declared field')
        for field in self.program.extended_spec.schema.fields:
            decode_value(field.field_type, row[field.name])
        if not self.program.extended_spec.evaluate(row):
            raise ValueError('row violates typed declared relation')
        return super().serialize(row, order)

class TypedEngine(SemanticByteEngine):

    def __init__(self, program):
        super().__init__(program)
        self.serializer = TypedSerializer(program)

    def finish_typed(self, state):
        row = self.finish(state)
        return {n: decode_value(self.program.extended_spec.schema.field(n).field_type, v) for n, v in row.items()}

class TypedMasker(LegacyMasker):

    def _numeric_walk(self, node, state, codec, allowed, ids):
        s = state.serializer
        language, _ = self._cached(self._languages, (codec.scale, codec.domain.intervals, s.value_prefix, allowed.intervals), lambda: numeric_language(codec, s.value_prefix, allowed))
        partial, exits = self._numeric_plan(node, language)
        ids.update(partial)
        self._consume_exits(exits, state, ids)

    def _consume_exits(self, exits, state, ids):
        if len(state.serializer.committed) + 1 >= len(self.engine.serializer.codecs):
            return
        for child, payload in exits:
            if 32 not in child.children:
                ids.update(child.terminal_ids)
            else:
                self._walk(child, self.engine.feed_bytes(state, payload), ids)

    def _text_walk(self, node, state, codec, allowed, ids):
        prefix = state.serializer.value_prefix
        exits = []
        for word in codec.candidate_spellings(prefix, allowed):
            suffix, cursor = (word[len(prefix):], node)
            for byte in suffix:
                cursor = cursor.children.get(byte)
                if cursor is None:
                    break
                ids.update(cursor.terminal_ids)
            else:
                comma = cursor.children.get(44)
                if comma is not None:
                    exits.append((comma, suffix + b','))
        self._consume_exits(exits, state, ids)

    def _walk(self, node, state, ids):
        if state.serializer.phase == 'header':
            return super()._walk(node, state, ids)
        _, codec, allowed = self.engine._context(state)
        if type(codec) is NumericCodec:
            return super()._walk(node, state, ids)
        if type(codec) is NullableNumericCodec:
            self._numeric_walk(node, state, codec.numeric, allowed.intersect(codec.numeric.domain), ids)
        self._text_walk(node, state, codec, allowed, ids)

    def legal_token_transitions(self, state):
        if state.serializer.phase == 'value':
            _, codec, _ = self.engine._context(state)
            if type(codec) is not NumericCodec:
                self.transition_builds += 1
                ids = set()
                self._walk(self._root, state, ids)
                if self.engine.can_end(state):
                    ids.add(self.eos_token_id)
                return LazyTransitions(self, state, tuple(sorted(ids)))
        return super().legal_token_transitions(state)

    def selected_successor(self, state, payload):
        if state.serializer.phase == 'value' and b',' not in payload:
            s = state.serializer
            successor = SemanticState(SerializerState('value', b'', s.current_field, s.value_prefix + payload, s.committed), state.residual)
            self.engine._context(successor)
            return successor
        return super().selected_successor(state, payload)

def make_masker(program, adapter):
    if type(program) is ExtendedProgram:
        return TypedMasker(TypedEngine(program), adapter)
    return LegacyMasker(SemanticByteEngine(program), adapter)
