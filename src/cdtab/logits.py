from __future__ import annotations
from collections.abc import Iterable
from cdtab.errors import InvalidModelLogitsError, ZeroLegalModelMassError

def apply_exact_mask(logits, allowed_token_ids: Iterable[int]):
    try:
        import torch
    except ImportError as error:
        raise RuntimeError('PyTorch is required for runtime logit masking') from error
    if logits.ndim != 1:
        raise InvalidModelLogitsError('v0.1 requires one-dimensional batch=1 logits')
    if torch.isnan(logits).any() or torch.isposinf(logits).any():
        raise InvalidModelLogitsError('model logits contain NaN or positive infinity')
    allowed = tuple(sorted(set((int(token_id) for token_id in allowed_token_ids))))
    if not allowed:
        raise ZeroLegalModelMassError('the compiled language exposes no legal token')
    if allowed[0] < 0 or allowed[-1] >= logits.shape[0]:
        raise InvalidModelLogitsError('allowed token id lies outside the model vocabulary')
    index = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    if torch.isneginf(logits.index_select(0, index)).all():
        raise ZeroLegalModelMassError('the model assigns zero mass to every legal token')
    masked = torch.full_like(logits, float('-inf'))
    masked.index_copy_(0, index, logits.index_select(0, index))
    return masked
