"""Consume native tokens while enforcing complete-row EOS acceptance."""

from __future__ import annotations

from typing import Iterable

from cdtab.errors import InvalidTransitionError
from cdtab.logits import apply_exact_mask
from cdtab.serializer import SemanticState as OrderFlexibleSemanticState
from cdtab.tokenizer import OrderFlexibleTokenMasker


class OrderFlexibleDecodingSession:
    def __init__(
        self,
        masker: OrderFlexibleTokenMasker,
        *,
        prefix: bytes = b"",
        selected_order: Iterable[str] | None = None,
    ):
        self.masker = masker
        self.state: OrderFlexibleSemanticState = masker.engine.state_for_prefix(
            bytes(prefix), selected_order=selected_order
        )
        self.finished = False
        self.row: dict[str, int] | None = None

    def allowed_token_ids(self) -> tuple[int, ...]:
        if self.finished:
            raise InvalidTransitionError("decoding session has already finished")
        return self.masker.legal_token_ids(self.state)

    def mask_logits(self, logits):
        return apply_exact_mask(logits, self.allowed_token_ids())

    def consume(self, token_id: int) -> dict[str, int] | None:
        if self.finished:
            raise InvalidTransitionError("cannot consume a token after EOS")
        token_id = int(token_id)
        if token_id == self.masker.eos_token_id:
            if not self.masker.engine.can_end(self.state):
                raise InvalidTransitionError("EOS is illegal before a complete row")
            self.row = self.masker.engine.finish(self.state)
            self.finished = True
            return dict(self.row)
        self.state = self.masker.advance_token(self.state, token_id)
        return None
