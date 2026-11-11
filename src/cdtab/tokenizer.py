from __future__ import annotations
from dataclasses import dataclass, field
import hashlib
from cdtab.errors import InvalidTransitionError
from cdtab.tokenizer_base import TokenPayloadAdapter
from cdtab.token_contract import admit_adapter
from cdtab.resources import ResourceBudget
from cdtab.serializer import SemanticByteEngine as OrderFlexibleSemanticByteEngine, SemanticState as OrderFlexibleSemanticState

@dataclass
class _TrieNode:
    children: dict[int, '_TrieNode'] = field(default_factory=dict)
    terminal_ids: list[int] = field(default_factory=list)

class OrderFlexibleTokenMasker:

    def __init__(self, engine: OrderFlexibleSemanticByteEngine, adapter: TokenPayloadAdapter):
        self.engine = engine
        self.adapter = adapter
        self._root = _TrieNode()
        budget = ResourceBudget(engine.program._limits)
        with budget.phase('tokenizer_admission'):
            payloads, self.contract = admit_adapter(adapter, engine.serializer.required_bytes, budget)
            self.eos_token_id = adapter.eos_token_id
        with budget.phase('token_trie'):
            for token_id, payload in payloads.items():
                budget.check()
                if token_id == self.eos_token_id or payload is None:
                    continue
                node = self._root
                for byte in payload:
                    node = node.children.setdefault(byte, _TrieNode())
                node.terminal_ids.append(token_id)
        self.setup_diagnostics = budget.snapshot()
        self._cached_state = None
        self._cached_transitions = None
        self.transition_builds = 0
        self.transition_cache_hits = 0

    def legal_token_transitions(self, state: OrderFlexibleSemanticState) -> dict[int, OrderFlexibleSemanticState | None]:
        if state == self._cached_state and self._cached_transitions is not None:
            self.transition_cache_hits += 1
            return dict(self._cached_transitions)
        self.transition_builds += 1
        transitions: dict[int, OrderFlexibleSemanticState | None] = {}
        stack: list[tuple[_TrieNode, OrderFlexibleSemanticState]] = [(self._root, state)]
        while stack:
            node, cursor = stack.pop()
            for token_id in node.terminal_ids:
                transitions[token_id] = cursor
            legal_bytes = set(self.engine.legal_next_bytes(cursor))
            for byte, child in node.children.items():
                if byte not in legal_bytes:
                    continue
                stack.append((child, self.engine.feed_byte(cursor, byte)))
        if self.engine.can_end(state):
            transitions[self.eos_token_id] = None
        self._cached_state = state
        self._cached_transitions = tuple(sorted(transitions.items()))
        return dict(self._cached_transitions)

    def legal_token_ids(self, state: OrderFlexibleSemanticState) -> tuple[int, ...]:
        return tuple(self.legal_token_transitions(state))

    def advance_token(self, state: OrderFlexibleSemanticState, token_id: int) -> OrderFlexibleSemanticState:
        token_id = int(token_id)
        transitions = self.legal_token_transitions(state)
        if token_id not in transitions:
            raise InvalidTransitionError(f'token {token_id} is illegal in this live prefix')
        next_state = transitions[token_id]
        if token_id == self.eos_token_id:
            if next_state is not None or not self.engine.can_end(state):
                raise InvalidTransitionError('EOS transition is inconsistent')
            return state
        if next_state is None:
            raise InvalidTransitionError('ordinary token has no successor state')
        return next_state

    @property
    def fingerprint(self) -> str:
        material = self.engine.fingerprint + ':' + str(self.contract['adapter_fingerprint'])
        return hashlib.sha256(material.encode('ascii')).hexdigest()
__all__ = ['GPT2ByteBPEAdapter', 'OrderFlexibleTokenMasker', 'StaticTokenAdapter', 'TokenPayloadAdapter']
